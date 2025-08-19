"""
从动作分类结果（actions.csv）生成健康状态结果（health.csv）。

规则：
- 对每个 animal，找到 action == 'lame' 的连续片段；
- 仅当片段持续时长 >= 阈值（默认 1.0s）时，标记该片段内所有帧 health='lame'，否则为 'healthy'；
- 输出列：frame[, time_s], animal, health。

使用示例：
python /root/autodl-tmp/AAH/health_from_actions.py \
  --actions-csv /root/autodl-tmp/AAH/output/test_unhealthy_actions.csv \
  --output /root/autodl-tmp/AAH/output/test_unhealthy_health.csv \
  --video /root/autodl-tmp/AAH/test/test_unhealthy.mp4 \
  --lame-min-sec 0.3
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def _try_get_fps(video_path: Optional[str]) -> Optional[float]:
    """尝试读取视频帧率。

    /**
     * @param {str | None} video_path - 视频路径
     * @returns {number | None} 成功则返回 FPS，否则 None
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


def generate_health_csv(
    actions_csv: str,
    output_csv: str,
    video_path: Optional[str] = None,
    lame_min_sec: float = 0.3,
    bridge_gap_frames: int = 0,
) -> str:
    """从 actions.csv 生成逐帧健康状态 CSV。

    /**
     * @param {str} actions_csv - 输入动作 CSV（由 action_classify.py 生成）
     * @param {str} output_csv - 输出健康 CSV 路径
     * @param {str | None} video_path - 可选：视频路径，用于在缺少 time_s 时计算时长
     * @param {number} lame_min_sec - 连续 lame 片段最短时长（秒）
     * @returns {str} 输出 CSV 路径
     */
    """
    df = pd.read_csv(actions_csv)
    if df.empty:
        out = pd.DataFrame(columns=["frame", "time_s", "animal", "health"]) if "time_s" in df.columns else pd.DataFrame(columns=["frame", "animal", "health"]) 
        out.to_csv(output_csv, index=False)
        return output_csv

    has_time = "time_s" in df.columns
    fps = None if has_time else _try_get_fps(video_path)
    if not has_time and not fps:
        fps = 30.0

    outputs: List[pd.DataFrame] = []
    for animal, sub in df.groupby("animal"):
        sub = sub.sort_values("frame").reset_index(drop=True)
        acts = sub["action"].astype(str).values
        frames = sub["frame"].astype(int).values
        is_lame = (acts == "lame")
        segs = _contiguous_true_segments(is_lame)

        # 桥接相邻 lame 片段：若间隔帧数 <= bridge_gap_frames，则合并为一段
        merged_segs: List[Tuple[int, int]] = []
        if segs:
            cur_s, cur_e = segs[0]
            for s, e in segs[1:]:
                gap = s - cur_e - 1
                if gap <= int(max(0, bridge_gap_frames)):
                    # 合并：扩大当前段的右端
                    cur_e = e
                else:
                    merged_segs.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged_segs.append((cur_s, cur_e))
        else:
            merged_segs = []

        keep = np.zeros_like(is_lame, dtype=bool)
        if has_time:
            t = sub["time_s"].astype(float).values
            for s, e in merged_segs:
                dur = float(t[e] - t[s]) if e >= s else 0.0
                if dur >= float(lame_min_sec):
                    keep[s:e + 1] = True
        else:
            fps_eff = float(fps) if (fps is not None and fps > 0.1) else 30.0
            for s, e in merged_segs:
                dur = float((frames[e] - frames[s]) / fps_eff) if e >= s else 0.0
                if dur >= float(lame_min_sec):
                    keep[s:e + 1] = True

        health = np.where(keep, "lame", "healthy")
        out = pd.DataFrame({
            "frame": frames,
            "animal": animal,
            "health": health.astype(str),
        })
        if has_time:
            out.insert(1, "time_s", sub["time_s"].astype(float).values)
        outputs.append(out)

    res = pd.concat(outputs, axis=0, ignore_index=True) if outputs else pd.DataFrame(columns=["frame", "animal", "health"]) 
    res.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="从动作结果生成逐帧健康状态 CSV（按连续 lame 片段阈值）")
    parser.add_argument("--actions-csv", type=str, required=True, help="由 action_classify.py 生成的动作 CSV")
    parser.add_argument("--output", type=str, required=True, help="输出健康 CSV 路径")
    parser.add_argument("--video", type=str, default=None, help="可选：对应视频路径（在缺少 time_s 时用于获得 FPS）")
    parser.add_argument("--lame-min-sec", type=float, default=1.0, help="连续 lame 判定最短时长（秒）")
    parser.add_argument("--bridge-gap-frames", type=int, default=0, help="桥接相邻 lame 段的最大空隙帧数，<=该值则合并为一段")
    args = parser.parse_args()

    out = generate_health_csv(
        actions_csv=args.actions_csv,
        output_csv=args.output,
        video_path=args.video,
        lame_min_sec=float(args.lame_min_sec),
        bridge_gap_frames=int(args.bridge_gap_frames),
    )
    print(f"Saved health CSV to: {out}")


if __name__ == "__main__":
    main()


