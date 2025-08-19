"""
基于 DeepLabCut 3.0 导出的关键点 CSV，对每一帧进行基础动作（stand/walk/run/lame）分类。

使用示例：

python /root/autodl-tmp/AAH/action_classify.py \
  --csv /root/autodl-tmp/AAH/output/test_unhealthy_keypoints_det.csv \
  --output /root/autodl-tmp/AAH/output/test_unhealthy_actions.csv \
  --video /root/autodl-tmp/AAH/test/test_unhealthy.mp4 \
  --lame-lift-frac 0.03 \
  --lame-min-sec 0.3

"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cv2  # 可选：用于读取视频 FPS
except Exception:
    cv2 = None


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


@dataclass
class Point2D:
    x: float
    y: float


def _euclidean_distance(p1: Point2D, p2: Point2D) -> float:
    dx = _safe_float(p1.x) - _safe_float(p2.x)
    dy = _safe_float(p1.y) - _safe_float(p2.y)
    if math.isnan(dx) or math.isnan(dy):
        return float("nan")
    return float(math.hypot(dx, dy))


def _try_get_fps(video_path: Optional[str]) -> Optional[float]:
    """尝试从视频读取 FPS。

    /**
     * @param {str | None} video_path - 输入视频路径
     * @returns {number | None} 若成功则返回帧率，否则返回 None
     */
    """
    if not video_path:
        return None
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps and fps > 0.1:
            return float(fps)
        return None
    finally:
        cap.release()


def _read_dlc_csv(csv_path: str) -> pd.DataFrame:
    """读取 DLC3.0 导出的 CSV（4 行表头，多级列）。

    /**
     * @param {string} csv_path - DLC 导出的关键点 CSV 路径
     * @returns {pd.DataFrame} index 为帧号，columns 为 MultiIndex(scorer, individuals, bodyparts, coords)
     */
    """
    # DLC3.0: 第一行为 scorer，第二行为 individuals，第三行为 bodyparts，第四行为 coords
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
    # 兼容：若 index 不是 int，尝试转换
    try:
        df.index = df.index.astype(int)
    except Exception:
        pass
    return df


def _list_individuals(df: pd.DataFrame) -> List[str]:
    """列出所有个体 ID。"""
    return sorted({col[1] for col in df.columns})


def _get_part_columns(df: pd.DataFrame, individual: str, bodypart: str) -> Optional[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str], Tuple[str, str, str, str]]]:
    """获取某个体、关键点的 (x, y, likelihood) 三列列标。

    /**
     * @param {pd.DataFrame} df
     * @param {string} individual - 个体名称，例如 "animal0"
     * @param {string} bodypart - 关键点名称，例如 "back_middle"
     * @returns {[MultiIndex, MultiIndex, MultiIndex] | None}
     */
    """
    candidates = [c for c in df.columns if c[1] == individual and c[2] == bodypart]
    if not candidates:
        return None
    x_col = next((c for c in candidates if c[3] == "x"), None)
    y_col = next((c for c in candidates if c[3] == "y"), None)
    l_col = next((c for c in candidates if c[3] == "likelihood"), None)
    if x_col and y_col and l_col:
        return x_col, y_col, l_col
    return None


def _extract_point_series(
    df: pd.DataFrame,
    individual: str,
    bodypart: str,
    min_likelihood: float,
) -> Optional[pd.DataFrame]:
    """提取关键点的按帧坐标（低置信度置为 NaN）。

    /**
     * @param {pd.DataFrame} df - DLC 多级列 DataFrame
     * @param {string} individual - 个体 ID
     * @param {string} bodypart - 关键点名称
     * @param {number} min_likelihood - 最小置信度阈值
     * @returns {pd.DataFrame | None} 列包含 [x, y, likelihood]
     */
    """
    cols = _get_part_columns(df, individual, bodypart)
    if cols is None:
        return None
    x_col, y_col, l_col = cols
    out = pd.DataFrame(index=df.index)
    out["x"] = df[x_col].astype(float)
    out["y"] = df[y_col].astype(float)
    out["likelihood"] = df[l_col].astype(float)
    mask = out["likelihood"] < float(min_likelihood)
    out.loc[mask, ["x", "y"]] = np.nan
    return out


def _compose_torso_center(
    df: pd.DataFrame,
    individual: str,
    min_likelihood: float,
) -> pd.DataFrame:
    """按帧估计躯干中心点（优先使用 `back_middle`，否则使用 `back_base` & `back_end`，再否则 `neck_base` & `tail_base`）。

    /**
     * @param {pd.DataFrame} df - DLC 多级列 DataFrame
     * @param {string} individual - 个体 ID
     * @param {number} min_likelihood - 最小置信度阈值
     * @returns {pd.DataFrame} 列包含 [x, y]
     */
    """
    pri = _extract_point_series(df, individual, "back_middle", min_likelihood)
    if pri is not None:
        center = pri[["x", "y"]].copy()
    else:
        center = pd.DataFrame(index=df.index, columns=["x", "y"], data=np.nan)

    def mean_two(bp1: str, bp2: str) -> Optional[pd.DataFrame]:
        p1 = _extract_point_series(df, individual, bp1, min_likelihood)
        p2 = _extract_point_series(df, individual, bp2, min_likelihood)
        if p1 is None or p2 is None:
            return None
        tmp = pd.DataFrame(index=df.index)
        tmp["x"] = (p1["x"] + p2["x"]) / 2.0
        tmp["y"] = (p1["y"] + p2["y"]) / 2.0
        return tmp

    if center[["x", "y"]].isna().all(axis=None):
        alt = mean_two("back_base", "back_end")
        if alt is None:
            alt = mean_two("neck_base", "tail_base")
        if alt is not None:
            center.loc[:, ["x", "y"]] = alt.values

    return center


def _estimate_body_length(
    df: pd.DataFrame,
    individual: str,
    min_likelihood: float,
) -> pd.Series:
    """按帧估计身体长度（mm 不可得，使用像素）。优先 `neck_base`-`tail_base`，否则 `back_base`-`back_end`。

    /**
     * @param {pd.DataFrame} df
     * @param {string} individual
     * @param {number} min_likelihood
     * @returns {pd.Series} 每帧一个像素长度，NaN 用临近与全局中位数填充
     */
    """
    def dist_of(bp1: str, bp2: str) -> Optional[pd.Series]:
        p1 = _extract_point_series(df, individual, bp1, min_likelihood)
        p2 = _extract_point_series(df, individual, bp2, min_likelihood)
        if p1 is None or p2 is None:
            return None
        d = np.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)
        d[p1[["x", "y"]].isna().any(axis=1) | p2[["x", "y"]].isna().any(axis=1)] = np.nan
        return d

    d = dist_of("neck_base", "tail_base")
    if d is None or d.isna().all():
        d = dist_of("back_base", "back_end")
    if d is None:
        d = pd.Series(index=df.index, dtype=float)

    # 临近填充 + 全局中位数回填
    d = d.ffill().bfill()
    if d.isna().any():
        med = float(np.nanmedian(d.values)) if len(d) else np.nan
        d = d.fillna(med)
    return d


def _rolling_mean(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr
    series = pd.Series(arr)
    return series.rolling(win, min_periods=1, center=False).mean().to_numpy()


# 候选“足部/蹄”关键点名称（优先级从前到后）
_FOOT_BODYPART_CANDIDATES: List[str] = [
    "coronet", "hoof", "paw", "foot",
    "front_paw", "hind_paw", "front_foot", "hind_foot",
    "fetlock", "ankle"
]


def _detect_lame_frames_4level(
    df: pd.DataFrame,
    individual: str,
    min_likelihood: float = 0.5,
    fps: Optional[float] = None,
    lift_frac_threshold: float = 0.06,
    min_duration_s: float = 1.0,
) -> np.ndarray:
    """检测某个体的“单脚持续离地（lame）”帧。

    /**
     * @param {pd.DataFrame} df - DLC 3.0 多级列 DataFrame（scorer/individuals/bodyparts/coords）
     * @param {string} individual - 个体 ID
     * @param {number} min_likelihood - 最小置信度阈值
     * @param {number | None} fps - 视频帧率，若为空则以 30 估计
     * @param {number} lift_frac_threshold - 抬起阈值（相对体长比例），默认 0.06
     * @param {number} min_duration_s - 最短持续时长（秒），默认 1.0
     * @returns {np.ndarray} 与帧数等长的布尔数组
     */
    """
    n = len(df.index)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    # 选取可用的足部关键点
    foot_series: Optional[pd.DataFrame] = None
    chosen_bp: Optional[str] = None
    for bp in _FOOT_BODYPART_CANDIDATES:
        s = _extract_point_series(df, individual, bp, min_likelihood)
        if s is None:
            continue
        valid = (~s[["x", "y"]].isna().any(axis=1)) & (s["likelihood"] >= float(min_likelihood))
        if valid.sum() >= max(10, int(0.05 * n)):
            foot_series = s
            chosen_bp = bp
            break

    if foot_series is None:
        return np.zeros((n,), dtype=bool)

    y_coro = foot_series["y"].to_numpy(dtype=float)
    l_coro = foot_series["likelihood"].to_numpy(dtype=float)
    valid_mask = (~np.isnan(y_coro)) & (~np.isnan(l_coro)) & (l_coro >= float(min_likelihood))
    if valid_mask.sum() < max(10, int(0.05 * n)):
        return np.zeros((n,), dtype=bool)

    # 地面基线 y：使用较高分位数（像素坐标系 y 向下增大）
    baseline_y = float(np.nanpercentile(y_coro[valid_mask], 85))

    # 体长序列
    body_len = _estimate_body_length(df, individual, min_likelihood).to_numpy(dtype=float)

    # 抬起幅度（相对体长）
    with np.errstate(invalid="ignore", divide="ignore"):
        lift_frac = (baseline_y - y_coro) / body_len

    lifted = (lift_frac > float(lift_frac_threshold)) & valid_mask

    # 连续时长约束
    fps_eff = float(fps) if (fps is not None and fps > 0.1) else 30.0
    min_len = max(1, int(round(float(min_duration_s) * fps_eff)))
    out = np.zeros((n,), dtype=bool)
    if lifted.any():
        idx = np.where(lifted)[0]
        start = idx[0]
        prev = idx[0]
        for k in range(1, len(idx)):
            if idx[k] == prev + 1:
                prev = idx[k]
            else:
                if (prev - start + 1) >= min_len:
                    out[start:prev + 1] = True
                start = idx[k]
                prev = idx[k]
        if (prev - start + 1) >= min_len:
            out[start:prev + 1] = True
    return out


def _list_bodyparts_for_individual(df: pd.DataFrame, individual: str) -> List[str]:
    """返回某个体的全部 bodyparts 名称列表。"""
    names = []
    for c in df.columns:
        try:
            if len(c) == 4 and c[1] == individual and c[3] == "x":
                names.append(str(c[2]))
        except Exception:
            continue
    return sorted(list(set(names)))


def _find_feet_bodyparts(df: pd.DataFrame, individual: str) -> List[str]:
    """从列名中推测四只脚的关键点名（尽量选取前左、前右、后左、后右）。

    规则：
    - 先筛选包含足部关键词的 bodyparts
    - 尝试按类别映射到 FL/FR/HL/HR 四类
    - 若不足四类，则从候选中补齐至最多 4 个（按有效帧数优先）
    """
    import re

    bodyparts = _list_bodyparts_for_individual(df, individual)
    foot_like = [bp for bp in bodyparts if any(tok in bp.lower() for tok in [
        "coronet", "hoof", "paw", "foot", "fetlock", "ankle"
    ])]

    def valid_count(bp: str) -> int:
        s = _extract_point_series(df, individual, bp, min_likelihood=0.0)
        if s is None:
            return 0
        v = (~s[["x", "y"]].isna().any(axis=1)).sum()
        return int(v)

    # 分类映射
    def category_of(name: str) -> Optional[str]:
        n = name.lower()
        # 规范化
        n = n.replace("-", "_").replace(" ", "_")
        # 前/后
        front = ("front" in n) or ("fore" in n) or bool(re.search(r"(^|_)f(r|l)?(_|$)", n))
        hind = ("hind" in n) or ("rear" in n) or ("back" in n) or bool(re.search(r"(^|_)h(r|l)?(_|$)", n))
        # 左/右
        left = ("left" in n) or bool(re.search(r"(^|_)(lf|fl|hl|lh|l)(_|$)", n))
        right = ("right" in n) or bool(re.search(r"(^|_)(rf|fr|hr|rh|r)(_|$)", n))

        if front and left:
            return "FL"
        if front and right:
            return "FR"
        if hind and left:
            return "HL"
        if hind and right:
            return "HR"
        return None

    chosen: Dict[str, str] = {}
    for bp in foot_like:
        cat = category_of(bp)
        if cat and cat not in chosen:
            chosen[cat] = bp
        if len(chosen) == 4:
            break

    # 若不足四个，用剩余 foot_like 按有效帧数排序补齐
    already = set(chosen.values())
    remain = [bp for bp in foot_like if bp not in already]
    remain = sorted(remain, key=valid_count, reverse=True)
    for bp in remain:
        if len(chosen) >= 4:
            break
        chosen[f"X{len(chosen)}"] = bp

    # 返回最多 4 个
    return list(chosen.values())[:4]


def _detect_lame_frames_by_feet_mean(
    df: pd.DataFrame,
    individual: str,
    min_likelihood: float = 0.5,
    fps: Optional[float] = None,
    lift_frac_threshold: float = 0.06,
    dominance_frac_threshold: float = 0.02,
    min_duration_s: float = 1.0,
    smooth_win_frames: int = 5,
) -> np.ndarray:
    """基于“四只脚相对均值”的方法检测 lame：

    步骤：
    - 选取四只脚关键点，逐帧计算 y 的均值 mean_y
    - 对每只脚计算 diff_i = mean_y - y_i（y 越小越靠上，抬起时 diff 越大）
    - 若某帧存在唯一的 top1，且：
        top1 >= lift_frac_threshold * body_length 且
        (top1 - top2) >= dominance_frac_threshold * body_length
      则认为该帧存在“脚抬起”
    - 满足持续时长 min_duration_s 的连续帧段记为 lame

    /**
     * @param {pd.DataFrame} df
     * @param {string} individual
     * @param {number} min_likelihood
     * @param {number | None} fps
     * @param {number} lift_frac_threshold
     * @param {number} dominance_frac_threshold - top1 与次大值的最小差额（体长比例）
     * @param {number} min_duration_s
     * @returns {np.ndarray}
     */
    """
    n = len(df.index)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    bps = _find_feet_bodyparts(df, individual)
    if len(bps) < 2:
        # 脚点不足，返回全 False（上层可选择回退策略）
        return np.zeros((n,), dtype=bool)

    ys: List[np.ndarray] = []
    ls: List[np.ndarray] = []
    for bp in bps:
        s = _extract_point_series(df, individual, bp, min_likelihood)
        if s is None:
            y = np.full((n,), np.nan, dtype=float)
            l = np.full((n,), np.nan, dtype=float)
        else:
            y = s["y"].to_numpy(dtype=float)
            l = s["likelihood"].to_numpy(dtype=float)
        # 平滑 y（对 NaN 友好）
        if int(smooth_win_frames) > 1:
            y = _rolling_mean(y, int(smooth_win_frames))
        ys.append(y)
        ls.append(l)

    Y = np.stack(ys, axis=1)  # (n, m)
    L = np.stack(ls, axis=1)
    valid = (~np.isnan(Y)) & (~np.isnan(L)) & (L >= float(min_likelihood))
    # 至少 2 个脚有效时才计算均值
    valid_rows = (valid.sum(axis=1) >= 2)

    # 按行计算均值，避免空切片告警
    counts = valid.sum(axis=1).astype(float)
    sum_y = np.nansum(np.where(valid, Y, 0.0), axis=1)
    mean_y = np.where(counts > 0.0, sum_y / counts, np.nan)
    body_len = _estimate_body_length(df, individual, min_likelihood).to_numpy(dtype=float)

    # diff = mean_y - yi
    diff = mean_y[:, None] - Y
    # 对无效位置填充 -inf，确保 argmax 安全
    diff_masked = np.where(valid, diff, -np.inf)

    # 判定：top1 与 top2（均使用 -inf 填充避免全 NaN 导致异常）
    idx_top1 = np.argmax(diff_masked, axis=1)
    top1 = diff_masked[np.arange(n), idx_top1]
    diff_second = diff_masked.copy()
    diff_second[np.arange(n), idx_top1] = -np.inf
    top2 = np.max(diff_second, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        cond_lift = (top1 / body_len) >= float(lift_frac_threshold)
        cond_dom = ((top1 - top2) / body_len) >= float(dominance_frac_threshold)

    # 至少需要两只脚有效以比较 top1 与 top2
    valid_two = (valid.sum(axis=1) >= 2)
    lifted = cond_lift & cond_dom & valid_rows & valid_two & np.isfinite(top1) & np.isfinite(top2)

    # 连续时长约束
    fps_eff = float(fps) if (fps is not None and fps > 0.1) else 30.0
    min_len = max(1, int(round(float(min_duration_s) * fps_eff)))
    out = np.zeros((n,), dtype=bool)
    if lifted.any():
        idx = np.where(lifted)[0]
        start = idx[0]
        prev = idx[0]
        for k in range(1, len(idx)):
            if idx[k] == prev + 1:
                prev = idx[k]
            else:
                if (prev - start + 1) >= min_len:
                    out[start:prev + 1] = True
                start = idx[k]
                prev = idx[k]
        if (prev - start + 1) >= min_len:
            out[start:prev + 1] = True
    return out


def classify_actions(
    df: pd.DataFrame,
    min_likelihood: float = 0.5,
    smoothing_window: int = 5,
    move_threshold_frac: float = 0.02,
    run_threshold_frac: float = 0.08,
    fps: Optional[float] = None,
    lame_lift_frac_threshold: float = 0.06,
    lame_dominance_frac_threshold: float = 0.02,
    lame_min_duration_s: float = 1.0,
) -> pd.DataFrame:
    """对每个个体逐帧判定 stand/walk/run/lame。

    规则：
    - 计算躯干中心逐帧位移（像素）并进行滑动平均平滑。
    - 以身体长度（像素）为尺度，得到归一化速度 v_norm = speed_px / body_len_px。
    - v_norm < move_threshold_frac -> stand
      move_threshold_frac <= v_norm < run_threshold_frac -> walk
      v_norm >= run_threshold_frac -> run
    - 额外：若检测到“单脚持续离地（lame）”片段，则该时段动作置为 lame 覆盖上述结果。

    /**
     * @param {pd.DataFrame} df - DLC 多级列 DataFrame
     * @param {number} min_likelihood - 最小置信度阈值，低于此阈值的关键点将置为 NaN
     * @param {number} smoothing_window - 速度平滑窗口（帧）
     * @param {number} move_threshold_frac - 站立与行走分界阈值（相对于体长/帧）
     * @param {number} run_threshold_frac - 行走与奔跑分界阈值（相对于体长/帧）
     * @param {number | None} fps - 帧率，用于 lame 持续时长判断
     * @param {number} lame_lift_frac_threshold - lame 判定抬起阈值（相对体长比例）
     * @param {number} lame_min_duration_s - lame 判定的最短持续秒数
     * @returns {pd.DataFrame} 列包含 [frame, animal, speed_px, body_length_px, speed_norm_body, action]
     */
    """
    individuals = _list_individuals(df)
    results: List[pd.DataFrame] = []

    for ind in individuals:
        center = _compose_torso_center(df, ind, min_likelihood)
        body_len = _estimate_body_length(df, ind, min_likelihood)

        # 逐帧速度（像素）
        dx = center["x"].diff().to_numpy()
        dy = center["y"].diff().to_numpy()
        speed = np.sqrt(dx ** 2 + dy ** 2)
        # 缺失补 0（首帧）
        if len(speed) > 0 and np.isnan(speed[0]):
            speed[0] = 0.0
        # 平滑
        speed_smooth = _rolling_mean(speed, int(smoothing_window))
        body_len_np = body_len.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            v_norm = np.where(body_len_np > 1e-6, speed_smooth / body_len_np, np.nan)

        action = np.full_like(v_norm, fill_value="", dtype=object)
        action[v_norm < float(move_threshold_frac)] = "stand"
        action[(v_norm >= float(move_threshold_frac)) & (v_norm < float(run_threshold_frac))] = "walk"
        action[v_norm >= float(run_threshold_frac)] = "run"

        # lame 检测并覆盖动作
        lame_flags = _detect_lame_frames_by_feet_mean(
            df=df,
            individual=ind,
            min_likelihood=float(min_likelihood),
            fps=fps,
            lift_frac_threshold=float(lame_lift_frac_threshold),
            dominance_frac_threshold=float(lame_dominance_frac_threshold),
            min_duration_s=float(lame_min_duration_s),
            smooth_win_frames=int(smoothing_window),
        )
        # 若基于四足方法无法判定，则回退到单足-基线方法
        if not np.any(lame_flags):
            lame_flags = _detect_lame_frames_4level(
                df=df,
                individual=ind,
                min_likelihood=float(min_likelihood),
                fps=fps,
                lift_frac_threshold=float(lame_lift_frac_threshold),
                min_duration_s=float(lame_min_duration_s),
            )
        action[lame_flags] = "lame"

        out = pd.DataFrame({
            "frame": df.index.values,
            "animal": ind,
            "speed_px": speed_smooth,
            "body_length_px": body_len_np,
            "speed_norm_body": v_norm,
            "action": action,
        })
        results.append(out)

    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    return pd.DataFrame(columns=["frame", "animal", "speed_px", "body_length_px", "speed_norm_body", "action"])


def main() -> None:
    parser = argparse.ArgumentParser(description="按帧动作分类（stand/walk/run/lame）")
    parser.add_argument("--csv", type=str, required=True, help="DLC 导出的关键点 CSV（4 级表头）")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径（默认与输入同目录，添加 _actions 后缀）")
    parser.add_argument("--video", type=str, default=None, help="可选：对应视频路径，用于写入 time_s")
    parser.add_argument("--min-likelihood", type=float, default=0.5, help="最小置信度阈值")
    parser.add_argument("--smooth-win", type=int, default=5, help="速度平滑窗口（帧）")
    parser.add_argument("--move-thr", type=float, default=0.02, help="站立/行走阈值（体长/帧）")
    parser.add_argument("--run-thr", type=float, default=0.08, help="行走/奔跑阈值（体长/帧）")
    parser.add_argument("--lame-lift-frac", type=float, default=0.06, help="lame 判定抬起阈值（相对体长比例）")
    parser.add_argument("--lame-min-sec", type=float, default=1.0, help="lame 判定的最短持续时长（秒）")
    parser.add_argument("--lame-dominance-frac", type=float, default=0.02, help="top1 与 top2 的差额阈值（相对体长比例），用于四足与均值差方法")
    args = parser.parse_args()

    csv_path = args.csv
    out_path = args.output
    if out_path is None:
        stem = Path(csv_path).stem.replace("_keypoints_det", "")
        out_path = str(Path(csv_path).with_name(f"{stem}_actions.csv"))

    df = _read_dlc_csv(csv_path)

    # 先尝试读取 FPS（用于 lame 持续时长判定以及可选 time_s 写入）
    fps = _try_get_fps(args.video)
    res = classify_actions(
        df,
        min_likelihood=float(args.min_likelihood),
        smoothing_window=int(args.smooth_win),
        move_threshold_frac=float(args.move_thr),
        run_threshold_frac=float(args.run_thr),
        fps=fps,
        lame_lift_frac_threshold=float(getattr(args, "lame_lift_frac", 0.06)),
        lame_dominance_frac_threshold=float(getattr(args, "lame_dominance_frac", 0.02)),
        lame_min_duration_s=float(getattr(args, "lame_min_sec", 0.3)),
    )

    # 可选写入时间戳
    if fps and fps > 0:
        res["time_s"] = res["frame"].astype(float) / float(fps)
        # 调整列顺序
        res = res[["frame", "time_s", "animal", "action", "speed_px", "body_length_px", "speed_norm_body"]]
    else:
        res = res[["frame", "animal", "action", "speed_px", "body_length_px", "speed_norm_body"]]

    os.makedirs(str(Path(out_path).parent), exist_ok=True)
    res.to_csv(out_path, index=False)

    print(f"Saved actions to: {out_path}")


if __name__ == "__main__":
    main()


