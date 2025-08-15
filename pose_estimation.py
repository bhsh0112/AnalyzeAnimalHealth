import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import pandas as pd
import numpy as np
import yaml


def _safe_read_csv_with_multiindex(csv_path: str) -> pd.DataFrame:
    """
    使用多级表头读取 DeepLabCut 导出的关键点 CSV，兼容 2.3/3.x 不同层级。

    @param csv_path str: 关键点预测 CSV 路径（例如 `*_predictions.csv`）
    @returns pd.DataFrame: 列为 MultiIndex 的 DataFrame（层级通常为 3 或 4）
    """
    # 优先尝试 4 层（scorer, individual, bodypart, coord）
    try:
        return pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
    except Exception:
        pass

    # 回退到 3 层（scorer, bodypart, coord）
    try:
        return pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    except Exception as exc:
        raise RuntimeError(f"无法读取 CSV: {csv_path}. 错误: {exc}") from exc


def _detect_schema(df: pd.DataFrame) -> Dict[str, bool]:
    """
    识别表头结构是否包含个体（individual）这一层级。

    @param df pd.DataFrame: DLC 关键点 DataFrame
    @returns Dict[str, bool]: {"has_individuals": bool}
    """
    nlevels = df.columns.nlevels
    has_individuals = nlevels == 4
    return {"has_individuals": has_individuals}


def _list_individuals(df: pd.DataFrame) -> List[str]:
    """
    列出数据中的个体 ID。若无个体层级，则返回 ["single"]。

    @param df pd.DataFrame
    @returns List[str]
    """
    if df.columns.nlevels == 4:
        # (scorer, individual, bodypart, coord)
        individuals = sorted({col[1] for col in df.columns})
        return individuals
    return ["single"]


def _list_bodyparts(df: pd.DataFrame, individual: str) -> List[str]:
    """
    列出某个个体（或单动物）的全部关键点名称。

    @param df pd.DataFrame
    @param individual str: 个体 ID 或 "single"
    @returns List[str]
    """
    if df.columns.nlevels == 4:
        return sorted({col[2] for col in df.columns if col[1] == individual})
    # 3 层结构: (scorer, bodypart, coord)
    return sorted({col[1] for col in df.columns})


def _get_series(
    df: pd.DataFrame,
    individual: str,
    bodypart: str,
    coord: str,
    likelihood_cutoff: float,
) -> pd.Series:
    """
    取出某个关键点的 x/y/likelihood 序列，并将低于阈值的坐标置为 NaN。

    @param df pd.DataFrame
    @param individual str: 个体 ID 或 "single"
    @param bodypart str: 关键点名称
    @param coord str: "x" | "y" | "likelihood"
    @param likelihood_cutoff float: 可信度阈值
    @returns pd.Series
    """
    series: Optional[pd.Series] = None
    if df.columns.nlevels == 4:
        # 假定层级顺序为 (scorer, individual, bodypart, coord)
        matches = [c for c in df.columns if c[1] == individual and c[2] == bodypart and c[3] == coord]
        if matches:
            series = df[matches[0]]
    else:
        # 假定层级顺序为 (scorer, bodypart, coord)
        matches = [c for c in df.columns if c[1] == bodypart and c[2] == coord]
        if matches:
            series = df[matches[0]]

    if series is None:
        return pd.Series(index=df.index, data=np.nan)

    if coord in ("x", "y"):
        # 遮蔽低置信度的坐标
        like = _get_series(df, individual, bodypart, "likelihood", likelihood_cutoff)
        series = series.mask(like < likelihood_cutoff)
    return series


def _compute_angle(
    ax: pd.Series,
    ay: pd.Series,
    bx: pd.Series,
    by: pd.Series,
    cx: pd.Series,
    cy: pd.Series,
) -> pd.Series:
    """
    计算以 B 点为顶点，向量 BA 与 BC 的夹角，单位为度。

    @param ax pd.Series: A.x
    @param ay pd.Series: A.y
    @param bx pd.Series: B.x
    @param by pd.Series: B.y
    @param cx pd.Series: C.x
    @param cy pd.Series: C.y
    @returns pd.Series: 每帧角度（度）
    """
    v1x = ax - bx
    v1y = ay - by
    v2x = cx - bx
    v2y = cy - by

    dot = v1x * v2x + v1y * v2y
    n1 = np.sqrt(v1x * v1x + v1y * v1y)
    n2 = np.sqrt(v2x * v2x + v2y * v2y)
    denom = n1 * n2
    cosang = dot / denom
    cosang = cosang.clip(-1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    ang[(n1 == 0) | (n2 == 0)] = np.nan
    return ang


def _compute_center(
    parts_xy: List[Tuple[pd.Series, pd.Series]],
) -> Tuple[pd.Series, pd.Series]:
    """
    计算若干关键点的平均中心（忽略 NaN）。

    @param parts_xy List[Tuple[Series, Series]]: 多个 (x, y) 序列
    @returns Tuple[Series, Series]: center_x, center_y
    """
    if not parts_xy:
        # 返回全 NaN
        idx = parts_xy[0][0].index if parts_xy else []
        return pd.Series(index=idx, data=np.nan), pd.Series(index=idx, data=np.nan)

    xs = [xy[0] for xy in parts_xy]
    ys = [xy[1] for xy in parts_xy]
    cx = pd.concat(xs, axis=1).mean(axis=1, skipna=True)
    cy = pd.concat(ys, axis=1).mean(axis=1, skipna=True)
    return cx, cy


def _compute_speed(cx: pd.Series, cy: pd.Series, fps: Optional[float]) -> pd.Series:
    """
    根据中心点序列计算速度。

    @param cx pd.Series: center.x
    @param cy pd.Series: center.y
    @param fps float|None: 帧率；若为 None 返回像素/帧，否则返回像素/秒
    @returns pd.Series
    """
    dx = cx.diff()
    dy = cy.diff()
    dist = np.sqrt(dx * dx + dy * dy)
    if fps is None:
        return dist  # px/frame
    return dist * fps  # px/s


def _load_angle_spec(yaml_path: Optional[str]) -> Dict[str, List[List[str]]]:
    """
    加载角度计算配置（可选）。

    YAML 示例：
    angles:
      - ["LShoulder", "LElbow", "LPaw"]
      - ["RShoulder", "RElbow", "RPaw"]
    center_parts: ["LShoulder", "RShoulder", "LHip", "RHip"]

    @param yaml_path str|None
    @returns Dict: {"angles": List[List[str]], "center_parts": List[str]}
    """
    if not yaml_path:
        return {"angles": [], "center_parts": []}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "angles": list(data.get("angles", []) or []),
        "center_parts": list(data.get("center_parts", []) or []),
    }


def _auto_center_parts(available_parts: List[str]) -> List[str]:
    """
    若未提供 center_parts，则基于常见命名自动选择一组中心关键点。

    @param available_parts List[str]
    @returns List[str]
    """
    candidates = [
        ["LShoulder", "RShoulder", "LHip", "RHip"],
        ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        ["spine1", "spine2", "spine3"],
    ]
    aset = set(available_parts)
    for group in candidates:
        if set(group) & aset:  # 至少存在 1 个
            return [p for p in group if p in aset]
    # 兜底：若没有常见命名，选取全部可用关键点的平均
    return available_parts[:]


def _compose_column_name(*parts: str) -> str:
    """
    组合安全的列名。

    @returns str
    """
    return ".".join([str(p) for p in parts if p is not None and p != ""]) \
        .replace(" ", "_")


def _normalize_name(name: str) -> str:
    """
    归一化关键点名称用于规则匹配（小写、移除分隔符）。

    @param name str
    @returns str
    """
    return name.strip().replace(" ", "").replace("-", "").replace("_", "").lower()


def _guess_angle_triplets(bodyparts: List[str]) -> List[List[str]]:
    """
    基于常见的四足/人形命名，猜测几个角度三元组，若不存在对应部位则跳过。

    @param bodyparts List[str]
    @returns List[List[str]]
    """
    parts_set = set(bodyparts)
    norm_map = {bp: _normalize_name(bp) for bp in bodyparts}

    def has_any(names: List[str]) -> Optional[str]:
        for bp in bodyparts:
            if norm_map[bp] in [ _normalize_name(n) for n in names ]:
                return bp
        return None

    triplets: List[List[str]] = []

    # 肢体：肩-肘-爪/腕
    for prefix in ["L", "R", "Left", "Right", "left", "right"]:
        shoulder = has_any([f"{prefix}Shoulder", f"{prefix}shoulder", f"{prefix}_shoulder"])  # type: ignore
        elbow = has_any([f"{prefix}Elbow", f"{prefix}elbow", f"{prefix}_elbow"])  # type: ignore
        paw = has_any([f"{prefix}Paw", f"{prefix}Wrist", f"{prefix}paw", f"{prefix}wrist", f"{prefix}_paw", f"{prefix}_wrist"])  # type: ignore
        if shoulder and elbow and paw:
            triplets.append([shoulder, elbow, paw])

    # 后肢：髋-膝-踝
    for prefix in ["L", "R", "Left", "Right", "left", "right"]:
        hip = has_any([f"{prefix}Hip", f"{prefix}hip", f"{prefix}_hip"])  # type: ignore
        knee = has_any([f"{prefix}Knee", f"{prefix}knee", f"{prefix}_knee"])  # type: ignore
        ankle = has_any([f"{prefix}Ankle", f"{prefix}ankle", f"{prefix}_ankle"])  # type: ignore
        if hip and knee and ankle:
            triplets.append([hip, knee, ankle])

    # 躯干：尾-躯干中点-头（若存在）
    head = has_any(["Head", "Snout", "Nose", "head", "snout", "nose"])  # type: ignore
    tail = has_any(["TailBase", "Tail", "tailbase", "tail"])  # type: ignore
    spine_candidates = [bp for bp in bodyparts if _normalize_name(bp).startswith("spine")]
    spine_mid = spine_candidates[len(spine_candidates)//2] if spine_candidates else None
    if head and spine_mid and tail:
        triplets.append([tail, spine_mid, head])

    # 去重
    unique = []
    seen = set()
    for t in triplets:
        key = tuple(t)
        if key not in seen:
            unique.append(t)
            seen.add(key)
    return unique


def _euclidean(ax: pd.Series, ay: pd.Series, bx: pd.Series, by: pd.Series) -> pd.Series:
    """
    计算两点间欧氏距离（逐帧）。

    @param ax pd.Series
    @param ay pd.Series
    @param bx pd.Series
    @param by pd.Series
    @returns pd.Series
    """
    dx = ax - bx
    dy = ay - by
    return np.sqrt(dx * dx + dy * dy)


def _rolling_mode(labels: pd.Series, window: int) -> pd.Series:
    """
    计算字符串标签的滑动众数（多数投票）。

    @param labels pd.Series[str]
    @param window int
    @returns pd.Series[str]
    """
    if window <= 1:
        return labels
    unique_vals = [v for v in pd.Series(labels.unique()).dropna().tolist() if isinstance(v, str)]
    to_int = {v: i + 1 for i, v in enumerate(sorted(unique_vals))}
    to_int["unknown"] = 0
    to_str = {i: v for v, i in to_int.items()}
    enc = labels.map(lambda x: to_int.get(x, 0))

    def mode_fn(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        vals, counts = np.unique(arr.astype(int), return_counts=True)
        return float(vals[np.argmax(counts)])

    rolled = enc.rolling(window=window, min_periods=1, center=True).apply(mode_fn, raw=True)
    return rolled.round().astype(int).map(lambda i: to_str.get(i, "unknown"))


def _classify_posture(
    df: pd.DataFrame,
    individual: str,
    likelihood_cutoff: float,
    cx: pd.Series,
    cy: pd.Series,
    stand_angle_threshold: float,
    bent_angle_threshold: float,
    smooth_window: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    基于关节角度与相对距离的启发式规则，对每帧进行姿态分类。

    类别："standing" | "lying" | "sitting" | "unknown"

    规则（启发式）：
    - 计算前肢（肩-肘-爪/腕）与后肢（髋-膝-踝/爪）角度；
      - 若所有可用肢体平均角度 >= stand_angle_threshold → 倾向站立。
      - 若平均角度 <= bent_angle_threshold 且四肢靠近身体中心 → 倾向趴卧。
      - 若前肢弯曲、后肢伸直或相反 → 倾向坐/蹲。

    @returns Tuple[Series, Series]: (raw_labels, smooth_labels)
    """
    bodyparts = _list_bodyparts(df, individual)
    available = set(bodyparts)

    def s(bp: str, c: str) -> pd.Series:
        return _get_series(df, individual, bp, c, likelihood_cutoff)

    head_like = [p for p in bodyparts if _normalize_name(p) in ("head", "snout", "nose")]
    tail_like = [p for p in bodyparts if "tail" in _normalize_name(p)]
    if head_like and tail_like:
        hl = _euclidean(s(head_like[0], "x"), s(head_like[0], "y"), s(tail_like[0], "x"), s(tail_like[0], "y"))
    else:
        shoulder_like = [p for p in bodyparts if "shoulder" in _normalize_name(p)]
        hip_like = [p for p in bodyparts if _normalize_name(p) == "hip" or _normalize_name(p).endswith("hip")]
        if shoulder_like and hip_like:
            hl = _euclidean(s(shoulder_like[0], "x"), s(shoulder_like[0], "y"), s(hip_like[0], "x"), s(hip_like[0], "y"))
        else:
            hl = pd.Series(index=df.index, data=np.nan)

    def pick(names: List[str]) -> Optional[str]:
        for n in names:
            if n in available:
                return n
        return None

    LShoulder = pick([n for n in bodyparts if _normalize_name(n) in ("lshoulder", "leftshoulder")])
    RShoulder = pick([n for n in bodyparts if _normalize_name(n) in ("rshoulder", "rightshoulder")])
    LElbow = pick([n for n in bodyparts if _normalize_name(n) in ("lelbow", "leftelbow")])
    RElbow = pick([n for n in bodyparts if _normalize_name(n) in ("relbow", "rightelbow")])
    LWrist = pick([n for n in bodyparts if _normalize_name(n) in ("lwrist", "lpaw", "leftwrist", "leftpaw")])
    RWrist = pick([n for n in bodyparts if _normalize_name(n) in ("rwrist", "rpaw", "rightwrist", "rightpaw")])

    LHip = pick([n for n in bodyparts if _normalize_name(n) in ("lhip", "lefthip")])
    RHip = pick([n for n in bodyparts if _normalize_name(n) in ("rhip", "righthip")])
    LKnee = pick([n for n in bodyparts if _normalize_name(n) in ("lknee", "leftknee")])
    RKnee = pick([n for n in bodyparts if _normalize_name(n) in ("rknee", "rightknee")])
    LAnkle = pick([n for n in bodyparts if _normalize_name(n) in ("lankle", "leftankle", "lpaw")])
    RAnkle = pick([n for n in bodyparts if _normalize_name(n) in ("rankle", "rightankle", "rpaw")])

    angles = []
    front_angles = []
    hind_angles = []
    if LShoulder and LElbow and LWrist:
        angles_LF = _compute_angle(s(LShoulder, "x"), s(LShoulder, "y"), s(LElbow, "x"), s(LElbow, "y"), s(LWrist, "x"), s(LWrist, "y"))
        angles.append(angles_LF)
        front_angles.append(angles_LF)
    if RShoulder and RElbow and RWrist:
        angles_RF = _compute_angle(s(RShoulder, "x"), s(RShoulder, "y"), s(RElbow, "x"), s(RElbow, "y"), s(RWrist, "x"), s(RWrist, "y"))
        angles.append(angles_RF)
        front_angles.append(angles_RF)
    if LHip and LKnee and LAnkle:
        angles_LH = _compute_angle(s(LHip, "x"), s(LHip, "y"), s(LKnee, "x"), s(LKnee, "y"), s(LAnkle, "x"), s(LAnkle, "y"))
        angles.append(angles_LH)
        hind_angles.append(angles_LH)
    if RHip and RKnee and RAnkle:
        angles_RH = _compute_angle(s(RHip, "x"), s(RHip, "y"), s(RKnee, "x"), s(RKnee, "y"), s(RAnkle, "x"), s(RAnkle, "y"))
        angles.append(angles_RH)
        hind_angles.append(angles_RH)

    def nanmean(series_list: List[pd.Series]) -> pd.Series:
        if not series_list:
            return pd.Series(index=df.index, data=np.nan)
        mat = pd.concat(series_list, axis=1)
        return mat.mean(axis=1, skipna=True)

    mean_all = nanmean(angles)
    mean_front = nanmean(front_angles)
    mean_hind = nanmean(hind_angles)

    paw_parts = [p for p in [LWrist, RWrist, LAnkle, RAnkle] if p is not None]
    paw_dists_norm = []
    for p in paw_parts:
        px = s(p, "x")
        py = s(p, "y")
        dist = _euclidean(px, py, cx, cy)
        paw_dists_norm.append(dist)
    mean_paw_to_center = nanmean(paw_dists_norm)
    norm = hl.copy()
    norm[norm < 1e-6] = np.nan
    rel_paw_center = mean_paw_to_center / norm

    raw = pd.Series(index=df.index, data="unknown", dtype=object)
    cond_stand = (mean_all >= stand_angle_threshold) & (rel_paw_center >= 0.25)
    raw[cond_stand] = "standing"
    cond_lying = (mean_all <= bent_angle_threshold) & (rel_paw_center < 0.25)
    raw[cond_lying] = "lying"
    cond_sit1 = (mean_front <= bent_angle_threshold) & (mean_hind >= stand_angle_threshold)
    cond_sit2 = (mean_hind <= bent_angle_threshold) & (mean_front >= stand_angle_threshold)
    raw[cond_sit1 | cond_sit2] = "sitting"

    smooth = _rolling_mode(raw, smooth_window)
    return raw, smooth


def estimate_pose_for_each_animal(
    csv_path: str,
    output_dir: Optional[str] = None,
    likelihood_cutoff: float = 0.1,
    fps: Optional[float] = None,
    angles_yaml: Optional[str] = None,
    enable_posture: bool = True,
    stand_angle_threshold: float = 140.0,
    bent_angle_threshold: float = 120.0,
    smooth_window: int = 5,
) -> Tuple[List[str], Path]:
    """
    读取关键点 CSV，对每一只动物计算姿态指标（方向、中心、速度、关节角度）。

    @param csv_path str: DLC 推理得到的 `*_predictions.csv`
    @param output_dir str|None: 输出目录；默认与 CSV 同目录
    @param likelihood_cutoff float: 关键点可信度阈值
    @param fps float|None: 视频帧率；提供则速度为像素/秒，否则为像素/帧
    @param angles_yaml str|None: 可选 YAML，定义角度三元组与中心点集合
    @returns Tuple[List[str], Path]: (individual 列表, 输出目录 Path)
    """
    df = _safe_read_csv_with_multiindex(csv_path)
    schema = _detect_schema(df)
    individuals = _list_individuals(df)
    angle_spec = _load_angle_spec(angles_yaml)

    out_dir = Path(output_dir or Path(csv_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_individual_outputs: Dict[str, Path] = {}
    summary_rows: List[Dict[str, object]] = []

    for individual in individuals:
        bodyparts = _list_bodyparts(df, individual)
        available = set(bodyparts)

        # 选择中心点集合
        center_parts = angle_spec.get("center_parts", []) or _auto_center_parts(bodyparts)
        center_pairs: List[Tuple[pd.Series, pd.Series]] = []
        for bp in center_parts:
            if bp in available:
                x = _get_series(df, individual, bp, "x", likelihood_cutoff)
                y = _get_series(df, individual, bp, "y", likelihood_cutoff)
                center_pairs.append((x, y))
        cx, cy = _compute_center(center_pairs)
        speed = _compute_speed(cx, cy, fps)

        # 尝试推断朝向（若有头/尾相关关键点）
        orientation_deg = pd.Series(index=df.index, data=np.nan)
        head_like = [p for p in bodyparts if p.lower() in ("head", "snout", "nose")]
        tail_like = [p for p in bodyparts if "tail" in p.lower()]
        if head_like and tail_like:
            hx = _get_series(df, individual, head_like[0], "x", likelihood_cutoff)
            hy = _get_series(df, individual, head_like[0], "y", likelihood_cutoff)
            tx = _get_series(df, individual, tail_like[0], "x", likelihood_cutoff)
            ty = _get_series(df, individual, tail_like[0], "y", likelihood_cutoff)
            dx = tx - hx
            dy = ty - hy
            orientation_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

        # 角度计算
        angle_triplets: List[List[str]] = angle_spec.get("angles", [])
        angle_cols: Dict[str, pd.Series] = {}
        for triplet in angle_triplets:
            if len(triplet) != 3:
                continue
            a, b, c = triplet
            if a not in available or b not in available or c not in available:
                continue
            ax = _get_series(df, individual, a, "x", likelihood_cutoff)
            ay = _get_series(df, individual, a, "y", likelihood_cutoff)
            bx = _get_series(df, individual, b, "x", likelihood_cutoff)
            by = _get_series(df, individual, b, "y", likelihood_cutoff)
            cx_ = _get_series(df, individual, c, "x", likelihood_cutoff)
            cy_ = _get_series(df, individual, c, "y", likelihood_cutoff)
            ang = _compute_angle(ax, ay, bx, by, cx_, cy_)
            col_name = _compose_column_name("angle", a, b, c)
            angle_cols[col_name] = ang

        # 姿态分类
        posture_raw = pd.Series(index=df.index, data="unknown", dtype=object)
        posture_smooth = pd.Series(index=df.index, data="unknown", dtype=object)
        if enable_posture:
            pr, ps = _classify_posture(
                df=df,
                individual=individual,
                likelihood_cutoff=likelihood_cutoff,
                cx=cx,
                cy=cy,
                stand_angle_threshold=stand_angle_threshold,
                bent_angle_threshold=bent_angle_threshold,
                smooth_window=smooth_window,
            )
            posture_raw = pr
            posture_smooth = ps

        # 组合结果表
        out_df = pd.DataFrame(index=df.index)
        out_df[_compose_column_name("center", "x")] = cx
        out_df[_compose_column_name("center", "y")] = cy
        out_df[_compose_column_name("speed", "px_per", "s" if fps is not None else "frame")] = speed
        out_df[_compose_column_name("orientation", "deg")] = orientation_deg
        out_df[_compose_column_name("posture", "raw")] = posture_raw
        out_df[_compose_column_name("posture", "smooth", f"w{smooth_window}")] = posture_smooth
        for k, v in angle_cols.items():
            out_df[k] = v

        stem = Path(csv_path).stem.replace("_predictions", "")
        indiv_tag = individual if individual != "single" else "animal"
        out_csv = out_dir / f"{stem}_pose_{indiv_tag}.csv"
        out_df.to_csv(out_csv)
        per_individual_outputs[individual] = out_csv

        # 汇总简单统计
        summary_rows.append(
            {
                "individual": individual,
                "n_frames": int(out_df.shape[0]),
                "speed_median": float(np.nanmedian(speed.values)) if len(out_df) else np.nan,
                "speed_mean": float(np.nanmean(speed.values)) if len(out_df) else np.nan,
                "orientation_circ_mean": float(
                    np.degrees(np.arctan2(
                        np.nanmean(np.sin(np.radians(orientation_deg))),
                        np.nanmean(np.cos(np.radians(orientation_deg))),
                    ))
                ) if np.isfinite(orientation_deg).any() else np.nan,
                "posture_standing_frac": float((posture_smooth == "standing").mean()) if len(out_df) else np.nan,
                "posture_lying_frac": float((posture_smooth == "lying").mean()) if len(out_df) else np.nan,
                "posture_sitting_frac": float((posture_smooth == "sitting").mean()) if len(out_df) else np.nan,
                "posture_unknown_frac": float((posture_smooth == "unknown").mean()) if len(out_df) else np.nan,
            }
        )

    # 写出汇总
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"{Path(csv_path).stem.replace('_predictions', '')}_pose_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return individuals, out_dir


def main() -> None:
    """
    从关键点 CSV 读取数据，对每个动物进行姿态估计并输出结果。

    命令行示例：
    python AAH/pose_estimation.py \
      --csv AAH/output/VAGotrdRsWk_predictions.csv \
      --output AAH/output \
      --fps 30 \
      --likelihood-cutoff 0.2 \
      --angles AAH/angles.yaml

    @returns None
    """
    parser = argparse.ArgumentParser(description="从关键点 CSV 进行多动物姿态估计")
    parser.add_argument("--csv", required=True, type=str, help="输入 *_predictions.csv")
    parser.add_argument("--output", type=str, default=None, help="输出目录，默认与 CSV 同目录")
    parser.add_argument("--fps", type=float, default=None, help="视频帧率；若不提供则速度单位为像素/帧")
    parser.add_argument("--likelihood-cutoff", type=float, default=0.1, help="关键点可信度阈值")
    parser.add_argument("--angles", type=str, default=None, help="可选 YAML，定义角度三元组与中心点")
    parser.add_argument("--list-bodyparts", action="store_true", help="仅列出 CSV 中的关键点名称后退出")
    parser.add_argument("--dump-angles-template", type=str, default=None, help="将基于关键点生成的 angles.yaml 模板写入到该路径，并退出")
    parser.add_argument("--no-posture", action="store_true", help="禁用姿态（站/趴/坐）分类输出")
    parser.add_argument("--stand-angle-threshold", type=float, default=140.0, help="站立判定的平均关节角度下限（度）")
    parser.add_argument("--bent-angle-threshold", type=float, default=120.0, help="弯曲判定的平均关节角度上限（度）")
    parser.add_argument("--smooth-window", type=int, default=5, help="姿态标签的多数投票平滑窗口大小（帧）")
    args = parser.parse_args()

    csv_path = args.csv
    output_dir = args.output
    likelihood_cutoff = float(args.likelihood_cutoff)
    fps = args.fps
    angles_yaml = args.angles
    enable_posture = not args.no_posture
    stand_angle_threshold = float(args.stand_angle_threshold)
    bent_angle_threshold = float(args.bent_angle_threshold)
    smooth_window = int(args.smooth_window)

    if not os.path.isfile(csv_path):
        print(f"找不到 CSV: {csv_path}", file=sys.stderr)
        sys.exit(2)

    # 特殊模式：列出关键点
    if args.list_bodyparts or args.dump_angles_template:
        df = _safe_read_csv_with_multiindex(csv_path)
        individuals = _list_individuals(df)
        by_indiv: Dict[str, List[str]] = {ind: _list_bodyparts(df, ind) for ind in individuals}

        if args.list_bodyparts:
            print("关键点列表：")
            for ind, bps in by_indiv.items():
                tag = ind if ind != "single" else "animal"
                print(f"- {tag}: {', '.join(bps)}")
            return

        if args.dump_angles_template:
            # 使用第一个个体的关键点做模板
            first_ind = individuals[0]
            parts = by_indiv[first_ind]
            center = _auto_center_parts(parts)
            guesses = _guess_angle_triplets(parts)
            tpl = {
                "center_parts": center,
                "angles": guesses,
            }
            out_path = Path(args.dump_angles_template)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(tpl, f, allow_unicode=True, sort_keys=False)
            print(f"已写出 angles 模板至: {out_path}")
            return

    individuals, out_dir = estimate_pose_for_each_animal(
        csv_path=csv_path,
        output_dir=output_dir,
        likelihood_cutoff=likelihood_cutoff,
        fps=fps,
        angles_yaml=angles_yaml,
        enable_posture=enable_posture,
        stand_angle_threshold=stand_angle_threshold,
        bent_angle_threshold=bent_angle_threshold,
        smooth_window=smooth_window,
    )

    print(
        f"已完成姿态估计。个体: {individuals}. 输出目录: {out_dir}")


if __name__ == "__main__":
    main()


