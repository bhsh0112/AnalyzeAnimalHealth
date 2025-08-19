"""
基于关键点检测与动作分类结果的健康状态判定与可视化。

使用示例：

python /root/autodl-tmp/AAH/health_visualize.py   --video /root/autodl-tmp/AAH/test/test_unhealthy.mp4   --keypoints-csv /root/autodl-tmp/AAH/output/test_unhealthy_keypoints_det.csv   --actions-csv /root/autodl-tmp/AAH/output/test_unhealthy_actions.csv   --output-video /root/autodl-tmp/AAH/output/test_unhealthy_health_vis.mp4   --health-lame-min-sec 0.3
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import zlib


# 颜色（BGR）
COLOR_POINT = (0, 215, 255)  # 金色
COLOR_EDGE = (0, 178, 255)   # 橙色
COLOR_TEXT = (0, 255, 0)     # 绿色
COLOR_WARN = (0, 0, 255)     # 红色
COLOR_BOX_GOOD = (0, 255, 0)
COLOR_BOX_BAD = (0, 0, 255)


# 关键点调色板（BGR，尽量区分明显）
COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (0, 0, 255),     # 红
    (0, 255, 0),     # 绿
    (255, 0, 0),     # 蓝
    (0, 255, 255),   # 黄
    (255, 0, 255),   # 品红
    (255, 255, 0),   # 青
    (0, 165, 255),   # 橙
    (180, 105, 255), # 粉
    (147, 20, 255),  # 紫
    (128, 128, 0),   # 橄榄
    (128, 0, 128),   # 紫罗兰
    (128, 0, 0),     # 暗红
    (0, 128, 128),   # 水鸭
    (0, 128, 255),   # 金黄
    (203, 192, 255), # 浅粉
    (50, 205, 50),   # 黄绿
    (139, 0, 0),     # 深红
    (0, 0, 139),     # 深蓝
    (0, 139, 139),   # 深青
    (238, 130, 238), # 紫罗兰红
]


def _get_bodypart_color(name: str) -> Tuple[int, int, int]:
    """返回某关键点名称的稳定专属颜色（BGR）。

    /**
     * @param {string} name - 关键点名称
     * @returns {[number,number,number]} BGR 颜色
     */
    """
    if not name:
        return COLOR_POINT
    idx = int(zlib.crc32(str(name).encode("utf-8")) % len(COLOR_PALETTE))
    return COLOR_PALETTE[idx]


# 可选骨架边（若 bodypart 存在则连接）
DEFAULT_EDGES: List[Tuple[str, str]] = [
    ("neck_base", "back_middle"),
    ("back_middle", "tail_base"),
    ("neck_base", "front_shoulder"),
    ("front_shoulder", "front_knee"),
    ("front_knee", "front_hoof"),
    ("tail_base", "hind_hip"),
    ("hind_hip", "hind_knee"),
    ("hind_knee", "hind_hoof"),
    ("withers", "elbow"),
    ("elbow", "knee"),
    ("knee", "coronet"),
    ("hock", "coronet"),
]


def _try_get_fps(video_path: Optional[str]) -> Optional[float]:
    """尝试读取视频帧率。

    /**
     * @param {str | None} video_path - 视频路径
     * @returns {float | None} 成功则返回 FPS，否则 None
     */
    """
    if not video_path:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps and fps > 0.1:
            return float(fps)
        return None
    finally:
        cap.release()


def _read_dlc3_csv(csv_path: str) -> pd.DataFrame:
    """读取 DLC3.0 导出的关键点 CSV（4 级表头）。

    /**
     * @param {str} csv_path - DLC 导出的关键点 CSV 路径
     * @returns {pd.DataFrame} index 为帧号，columns 为 MultiIndex(scorer, individuals, bodyparts, coords)
     */
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
    try:
        df.index = df.index.astype(int)
    except Exception:
        pass
    return df


def _list_individuals(df: pd.DataFrame) -> List[str]:
    """列出所有个体 ID。"""
    return sorted({col[1] for col in df.columns})


def _get_part_columns(
    df: pd.DataFrame,
    individual: str,
    bodypart: str,
) -> Optional[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str], Tuple[str, str, str, str]]]:
    """获取某个体、关键点的 (x, y, likelihood) 三列列标。

    /**
     * @param {pd.DataFrame} df
     * @param {string} individual - 个体名称，例如 "animal0"
     * @param {string} bodypart - 关键点名称
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


def _extract_points_for_frame(
    df: pd.DataFrame,
    individual: str,
    frame_idx: int,
    min_likelihood: float = 0.5,
) -> Dict[str, Tuple[float, float, float]]:
    """提取某帧指定个体的关键点集合。

    /**
     * @param {pd.DataFrame} df - DLC 多级列 DataFrame
     * @param {string} individual - 个体 ID
     * @param {number} frame_idx - 帧号（与 df.index 对齐）
     * @param {number} min_likelihood - 低于此阈值的点置 NaN
     * @returns {Object<string, [number, number, number]>}
     */
    """
    points: Dict[str, Tuple[float, float, float]] = {}
    # 收集该个体所有 bodyparts
    bodyparts = sorted({c[2] for c in df.columns if c[1] == individual and c[3] in ("x", "y")})
    if frame_idx not in df.index:
        return points
    for bp in bodyparts:
        cols = _get_part_columns(df, individual, bp)
        if cols is None:
            continue
        x_col, y_col, l_col = cols
        x = float(df.at[frame_idx, x_col]) if x_col in df.columns else float("nan")
        y = float(df.at[frame_idx, y_col]) if y_col in df.columns else float("nan")
        l = float(df.at[frame_idx, l_col]) if l_col in df.columns else float("nan")
        if not np.isnan(l) and l < float(min_likelihood):
            x, y = float("nan"), float("nan")
        points[bp] = (x, y, l)
    return points


def _compute_bbox(points: Dict[str, Tuple[float, float, float]], margin: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """由关键点估计检测框（取 xy 的 min/max）。

    /**
     * @param {Object} points - {name: (x, y, l)}
     * @param {number} margin - 外扩像素
     * @returns {[x1,y1,x2,y2] | None}
     */
    """
    xs: List[float] = []
    ys: List[float] = []
    for name, (x, y, l) in points.items():
        if np.isnan(x) or np.isnan(y):
            continue
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    x1 = int(max(0, min(xs) - margin))
    y1 = int(max(0, min(ys) - margin))
    x2 = int(max(xs) + margin)
    y2 = int(max(ys) + margin)
    return x1, y1, x2, y2


def _draw_skeleton(
    image: np.ndarray,
    points: Dict[str, Tuple[float, float, float]],
    edges: Sequence[Tuple[str, str]] = DEFAULT_EDGES,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """在图像上绘制关键点与骨架边。

    /**
     * @param {np.ndarray} image - BGR 图像
     * @param {Object} points - {name: (x, y, l)}
     * @param {Array<[string,string]>} edges - 骨架连接对
     * @param {number} conf_threshold - 置信度阈值
     * @returns {np.ndarray}
     */
    """
    canvas = image.copy()
    # 画边
    for a, b in edges:
        pa = points.get(a)
        pb = points.get(b)
        if not pa or not pb:
            continue
        if np.isnan(pa[0]) or np.isnan(pa[1]) or np.isnan(pb[0]) or np.isnan(pb[1]):
            continue
        if (not np.isnan(pa[2]) and pa[2] < conf_threshold) or (not np.isnan(pb[2]) and pb[2] < conf_threshold):
            continue
        cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), COLOR_EDGE, 2, cv2.LINE_AA)
    # 画点（每个关键点独立颜色）
    for name, (x, y, l) in points.items():
        if np.isnan(x) or np.isnan(y) or (not np.isnan(l) and l < conf_threshold):
            continue
        color_pt = _get_bodypart_color(name)
        cv2.circle(canvas, (int(x), int(y)), 3, color_pt, -1, cv2.LINE_AA)
    return canvas


def _draw_bbox(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    ok: bool,
    thickness: int = 2,
) -> np.ndarray:
    """绘制检测框（由关键点包围盒近似）。

    /**
     * @param {np.ndarray} image
     * @param {[number,number,number,number] | None} bbox - (x1,y1,x2,y2)
     * @param {bool} ok - True 用绿色，False 用红色
     * @param {number} thickness - 线宽
     * @returns {np.ndarray}
     */
    """
    if bbox is None:
        return image
    x1, y1, x2, y2 = bbox
    color = COLOR_BOX_GOOD if ok else COLOR_BOX_BAD
    canvas = image.copy()
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return canvas


def _overlay_text(
    image: np.ndarray,
    lines: Sequence[str],
    org: Tuple[int, int] = (10, 30),
    scale: float = 0.8,
    thickness: int = 2,
    color: Tuple[int, int, int] = COLOR_TEXT,
) -> np.ndarray:
    """在图像上叠加多行文本。"""
    canvas = image.copy()
    x, y = org
    for i, text in enumerate(lines):
        cv2.putText(canvas, str(text), (x, y + i * int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return canvas


def _draw_keypoints(
    image: np.ndarray,
    points: Dict[str, Tuple[float, float, float]],
    conf_threshold: float = 0.5,
    radius: int = 3,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """在图像上绘制关键点（不连线）。

    /**
     * @param {np.ndarray} image - BGR 图像
     * @param {Object} points - {name: (x, y, l)}
     * @param {number} conf_threshold - 置信度阈值
     * @param {number} radius - 圆点半径
     * @param {[number,number,number] | None} color - 若提供则统一颜色，否则按关键点名称着色
     * @returns {np.ndarray}
     */
    """
    canvas = image.copy()
    for name, (x, y, l) in points.items():
        if np.isnan(x) or np.isnan(y) or (not np.isnan(l) and l < conf_threshold):
            continue
        draw_color = color if (color is not None) else _get_bodypart_color(name)
        cv2.circle(canvas, (int(x), int(y)), int(radius), draw_color, -1, cv2.LINE_AA)
    return canvas


def _contiguous_true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """将布尔序列中的 True 段落记为 [start, end]（闭区间）。"""
    segs: List[Tuple[int, int]] = []
    idx = np.where(mask)[0]
    if idx.size == 0:
        return segs
    start = int(idx[0])
    prev = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k == prev + 1:
            prev = k
        else:
            segs.append((start, prev))
            start = k
            prev = k
    segs.append((start, prev))
    return segs


def _determine_health_status(
    actions_df: pd.DataFrame,
    fps: Optional[float],
    lame_min_sec: float = 1.0,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """根据动作结果判定健康状态：若存在连续 >= lame_min_sec 的 lame 则判为 lame。

    /**
     * @param {pd.DataFrame} actions_df - 列至少包含 frame, animal, action，可选 time_s
     * @param {number | None} fps - 帧率（当 actions 无 time_s 时使用）
     * @param {number} lame_min_sec - lame 连续最短秒数
     * @returns {[string, Object]} overall_status 与 per-animal 统计信息
     */
    """
    status = "healthy"
    stats: Dict[str, Dict[str, float]] = {}
    if actions_df.empty:
        return status, stats
    has_time = "time_s" in actions_df.columns
    for animal, sub in actions_df.groupby("animal"):
        sub = sub.sort_values("frame")
        acts = sub["action"].astype(str).values
        frames = sub["frame"].astype(int).values
        is_lame = (acts == "lame")
        segs = _contiguous_true_segments(is_lame)
        max_dur = 0.0
        if has_time:
            t = sub["time_s"].astype(float).values
            for s, e in segs:
                dur = float(t[e] - t[s]) if e >= s else 0.0
                max_dur = max(max_dur, dur)
        else:
            fps_eff = float(fps) if (fps is not None and fps > 0.1) else 30.0
            for s, e in segs:
                dur = float((frames[e] - frames[s]) / fps_eff) if e >= s else 0.0
                max_dur = max(max_dur, dur)
        stats[animal] = {
            "lame_segments": float(len(segs)),
            "max_lame_duration_s": float(max_dur),
        }
        if max_dur >= float(lame_min_sec):
            status = "lame"
    return status, stats


def _build_frame_health_map(
    actions_df: pd.DataFrame,
    fps: Optional[float],
    lame_min_sec: float = 1.0,
    bridge_gap_frames: int = 0,
) -> Dict[int, Dict[str, str]]:
    """基于动作结果构建逐帧健康状态映射：frame -> {animal: 'healthy'|'lame'}。

    /**
     * @param {pd.DataFrame} actions_df - 列至少包含 frame, animal, action，可选 time_s
     * @param {number | None} fps - 帧率（当 actions 无 time_s 时使用）
     * @param {number} lame_min_sec - lame 连续最短秒数
     * @param {number} bridge_gap_frames - 桥接相邻 lame 段的最大间隙帧数
     * @returns {Object} 健康状态映射
     */
    """
    health_map: Dict[int, Dict[str, str]] = {}
    if actions_df.empty:
        return health_map

    has_time = "time_s" in actions_df.columns
    fps_eff = float(fps) if (fps is not None and fps > 0.1) else 30.0

    def contiguous(mask: np.ndarray) -> List[Tuple[int, int]]:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []
        segs: List[Tuple[int, int]] = []
        s = int(idx[0])
        p = int(idx[0])
        for k in idx[1:]:
            k = int(k)
            if k == p + 1:
                p = k
            else:
                segs.append((s, p))
                s, p = k, k
        segs.append((s, p))
        return segs

    for animal, sub in actions_df.groupby("animal"):
        sub = sub.sort_values("frame").reset_index(drop=True)
        frames = sub["frame"].astype(int).values
        acts = sub["action"].astype(str).values
        is_lame = (acts == "lame")
        segs = contiguous(is_lame)

        # 桥接相邻 lame 段
        merged: List[Tuple[int, int]] = []
        if segs:
            cs, ce = segs[0]
            for s, e in segs[1:]:
                gap = s - ce - 1
                if gap <= int(max(0, bridge_gap_frames)):
                    ce = e
                else:
                    merged.append((cs, ce))
                    cs, ce = s, e
            merged.append((cs, ce))

        keep = np.zeros_like(is_lame, dtype=bool)
        if has_time and "time_s" in sub.columns:
            t = sub["time_s"].astype(float).values
            for s, e in merged:
                dur = float(t[e] - t[s]) if e >= s else 0.0
                if dur >= float(lame_min_sec):
                    keep[s:e + 1] = True
        else:
            for s, e in merged:
                dur = float((frames[e] - frames[s]) / fps_eff) if e >= s else 0.0
                if dur >= float(lame_min_sec):
                    keep[s:e + 1] = True

        # 写入 health_map
        for i, f in enumerate(frames):
            health_map.setdefault(int(f), {})[str(animal)] = "lame" if keep[i] else "healthy"

    return health_map


def visualize(
    video_path: str,
    keypoints_csv: str,
    actions_csv: str,
    output_video: str,
    min_likelihood: float = 0.5,
    bbox_margin: int = 10,
    text_scale: float = 0.8,
    text_thickness: int = 2,
    lame_min_sec: float = 1.0,
    bridge_gap_frames: int = 0,
) -> Tuple[str, str, Dict[str, Dict[str, float]]]:
    """生成可视化视频，并返回整体健康状态与统计。

    /**
     * @param {str} video_path - 输入视频
     * @param {str} keypoints_csv - DLC3.0 关键点 CSV（4 级表头）
     * @param {str} actions_csv - 每帧动作 CSV（由 action_classify.py 生成）
     * @param {str} output_video - 输出视频路径
     * @param {number} min_likelihood - 最小置信度阈值
     * @param {number} bbox_margin - 框外扩像素
     * @param {number} text_scale - 文本缩放
     * @param {number} text_thickness - 文本线宽
     * @param {number} lame_min_sec - 连续 lame 判定阈值（秒）
     * @returns {[str, str, Object]} (output_video_path, overall_status, per-animal 统计)
     */
    """
    # 读数据
    df = _read_dlc3_csv(keypoints_csv)
    actions = pd.read_csv(actions_csv)
    fps = _try_get_fps(video_path)
    overall_status, per_stats = _determine_health_status(actions, fps=fps, lame_min_sec=float(lame_min_sec))
    # 基于动作构建逐帧 health 映射
    frame_health_map = _build_frame_health_map(actions, fps=fps, lame_min_sec=float(lame_min_sec), bridge_gap_frames=int(bridge_gap_frames))

    # 准备视频 IO
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_eff = float(fps) if (fps is not None and fps > 0.1) else float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    os.makedirs(str(Path(output_video).parent), exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_eff,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"无法创建输出视频：{output_video}")

    # 将 actions 转为便于查询的映射：frame -> {animal: action}
    actions_map: Dict[int, Dict[str, str]] = {}
    for _, row in actions.iterrows():
        f = int(row.get("frame", 0))
        a = str(row.get("animal", "animal0"))
        act = str(row.get("action", ""))
        actions_map.setdefault(f, {})[a] = act

    individuals = _list_individuals(df)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # DLC CSV 的 index 直接为帧号，假设与视频对齐
        dlc_frame = current_frame if current_frame in df.index else None
        canvas = frame

        if dlc_frame is not None:
            for ind in individuals:
                pts = _extract_points_for_frame(df, ind, dlc_frame, min_likelihood=float(min_likelihood))
                # 仅绘制检测框与标签+健康状态
                health = frame_health_map.get(dlc_frame, {}).get(ind, "healthy")
                # 绘制关键点（仅点，按关键点名称多色）
                canvas = _draw_keypoints(canvas, pts, conf_threshold=float(min_likelihood), radius=3, color=None)
                bbox = _compute_bbox(pts, margin=int(bbox_margin))
                ok = (health != "lame")
                canvas = _draw_bbox(canvas, bbox, ok=ok, thickness=2)
                # 文本放置在检测框左上角
                if bbox is not None:
                    x1, y1, _, _ = bbox
                    label = actions_map.get(dlc_frame, {}).get(ind, "")
                    tag = f"{ind}: {health if health else 'healthy'}"
                    color = (COLOR_WARN if (health == 'lame') else COLOR_TEXT)
                    canvas = _overlay_text(canvas, [tag], org=(int(x1) + 4, int(y1) - 6 if y1 > 20 else int(y1) + 18), scale=float(text_scale), thickness=int(text_thickness), color=color)

        writer.write(canvas)
        current_frame += 1

        # 若 CSV 帧数小于视频帧数，直接继续写；若视频帧数未知则忽略
        if total_frames and current_frame >= total_frames:
            break

    cap.release()
    writer.release()
    return str(output_video), str(overall_status), per_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="根据动作判定健康状态并生成关键点/检测框可视化视频")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--keypoints-csv", type=str, required=True, help="DLC3.0 关键点 CSV（4 级表头）")
    parser.add_argument("--actions-csv", type=str, required=True, help="由 action_classify.py 生成的动作 CSV")
    parser.add_argument("--output-video", type=str, default=None, help="输出可视化视频路径（默认与输入视频同名添加 _health_vis 后缀）")
    parser.add_argument("--min-likelihood", type=float, default=0.5, help="最小关键点置信度阈值")
    parser.add_argument("--bbox-margin", type=int, default=10, help="检测框外扩像素")
    parser.add_argument("--text-scale", type=float, default=0.8, help="叠加文本缩放")
    parser.add_argument("--text-thickness", type=int, default=2, help="叠加文本线宽")
    parser.add_argument("--health-lame-min-sec", type=float, default=1.0, help="健康判定：lame 连续最短秒数阈值")
    parser.add_argument("--bridge-gap-frames", type=int, default=0, help="桥接相邻 lame 段的最大空隙帧数")
    args = parser.parse_args()

    video_path = args.video
    keypoints_csv = args.keypoints_csv
    actions_csv = args.actions_csv
    out_video = args.output_video
    if out_video is None:
        stem = Path(video_path).stem
        out_video = str(Path(video_path).with_name(f"{stem}_health_vis.mp4"))

    vis_path, status, stats = visualize(
        video_path=video_path,
        keypoints_csv=keypoints_csv,
        actions_csv=actions_csv,
        output_video=out_video,
        min_likelihood=float(args.min_likelihood),
        bbox_margin=int(args.bbox_margin),
        text_scale=float(args.text_scale),
        text_thickness=int(args.text_thickness),
        lame_min_sec=float(args.health_lame_min_sec),
        bridge_gap_frames=int(args.bridge_gap_frames),
    )

    print(f"Saved visualization to: {vis_path}")
    print(f"Overall health status: {status}")
    if stats:
        for k, v in stats.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()


