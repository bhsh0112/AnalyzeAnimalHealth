"""
基于 DeepLabCut 3.0 导出的关键点 CSV，对每一帧进行基础动作（stand/walk/run）分类。

使用示例：

python /root/autodl-tmp/AAH/action_classify.py \
  --csv /root/autodl-tmp/AAH/output/Sample1_keypoints_det.csv \
  --output /root/autodl-tmp/AAH/output/Sample1_actions.csv \
  --video /root/autodl-tmp/AAH/test/VAGotrdRsWk.mp4

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


def classify_actions(
    df: pd.DataFrame,
    min_likelihood: float = 0.5,
    smoothing_window: int = 5,
    move_threshold_frac: float = 0.02,
    run_threshold_frac: float = 0.08,
) -> pd.DataFrame:
    """对每个个体逐帧判定 stand/walk/run。

    规则：
    - 计算躯干中心逐帧位移（像素）并进行滑动平均平滑。
    - 以身体长度（像素）为尺度，得到归一化速度 v_norm = speed_px / body_len_px。
    - v_norm < move_threshold_frac -> stand
      move_threshold_frac <= v_norm < run_threshold_frac -> walk
      v_norm >= run_threshold_frac -> run

    /**
     * @param {pd.DataFrame} df - DLC 多级列 DataFrame
     * @param {number} min_likelihood - 最小置信度阈值，低于此阈值的关键点将置为 NaN
     * @param {number} smoothing_window - 速度平滑窗口（帧）
     * @param {number} move_threshold_frac - 站立与行走分界阈值（相对于体长/帧）
     * @param {number} run_threshold_frac - 行走与奔跑分界阈值（相对于体长/帧）
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
    parser = argparse.ArgumentParser(description="按帧动作分类（stand/walk/run）")
    parser.add_argument("--csv", type=str, required=True, help="DLC 导出的关键点 CSV（4 级表头）")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径（默认与输入同目录，添加 _actions 后缀）")
    parser.add_argument("--video", type=str, default=None, help="可选：对应视频路径，用于写入 time_s")
    parser.add_argument("--min-likelihood", type=float, default=0.5, help="最小置信度阈值")
    parser.add_argument("--smooth-win", type=int, default=5, help="速度平滑窗口（帧）")
    parser.add_argument("--move-thr", type=float, default=0.02, help="站立/行走阈值（体长/帧）")
    parser.add_argument("--run-thr", type=float, default=0.08, help="行走/奔跑阈值（体长/帧）")
    args = parser.parse_args()

    csv_path = args.csv
    out_path = args.output
    if out_path is None:
        stem = Path(csv_path).stem.replace("_keypoints_det", "")
        out_path = str(Path(csv_path).with_name(f"{stem}_actions.csv"))

    df = _read_dlc_csv(csv_path)
    res = classify_actions(
        df,
        min_likelihood=float(args.min_likelihood),
        smoothing_window=int(args.smooth_win),
        move_threshold_frac=float(args.move_thr),
        run_threshold_frac=float(args.run_thr),
    )

    # 可选写入时间戳
    fps = _try_get_fps(args.video)
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


