'''
示例
python /root/autodl-tmp/AAH/keypoints_detect.py \
  --video /root/autodl-tmp/AAH/test/VAGotrdRsWk.mp4 \
  --output /root/autodl-tmp/AAH/output \
  --pose /path/to/pose.pt \
  --detector /path/to/detector.pt \
  --local-only
'''

import os
import argparse
from pathlib import Path
import glob

# 禁用一切 GUI（在导入 deeplabcut 前设置）
os.environ["DLClight"] = "True"
os.environ.setdefault("MPLBACKEND", "Agg")


def _save_results_to_csv_and_cleanup(results: dict, video_path: str, dest_folder: str) -> None:
    """将推理结果 DataFrame 保存为 CSV，并删除多余的可视化视频和中间文件。

    /**
     * @param {dict} results - `video_inference_superanimal` 返回的 {video_path: pd.DataFrame} 映射
     * @param {str} video_path - 输入视频路径
     * @param {str} dest_folder - 输出目录
     * @returns {None}
     */
    """
    for _vid, df in results.items():
        stem = Path(video_path).stem
        csv_path = Path(dest_folder) / f"{stem}_keypoints_det.csv"
        df.to_csv(csv_path)

    stem = Path(video_path).stem
    patterns = [
        f"{stem}_*_labeled*.mp4",
        f"{stem}_*.h5",
        f"{stem}_*.json",
    ]
    for pat in patterns:
        for fpath in glob.glob(str(Path(dest_folder) / pat)):
            try:
                os.remove(fpath)
            except OSError:
                pass


def _run_dlc30(
    video_path: str,
    dest_folder: str,
    pose_ckpt: str | None = None,
    detector_ckpt: str | None = None,
) -> None:
    """使用 DLC 3.0 的 SuperAnimal API 运行推理。

    /**
     * @param {str} video_path - 输入视频
     * @param {str} dest_folder - 输出目录
     * @param {str | None} pose_ckpt - 可选，自定义姿态模型 checkpoint 路径（.pt）
     * @param {str | None} detector_ckpt - 可选，自定义检测器 checkpoint 路径（.pt）
     * @returns {None}
     */
    """
    try:
        from deeplabcut.modelzoo.video_inference import video_inference_superanimal
    except Exception as exc:
        raise SystemExit(
            "未检测到可用的 DeepLabCut 3.0（或导入失败）。请安装兼容版本的 DLC 3.x。原始错误：" + str(exc)
        )

    os.makedirs(dest_folder, exist_ok=True)
    results = video_inference_superanimal(
        videos=[video_path],
        superanimal_name="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        dest_folder=dest_folder,
        plot_bboxes=False,
        batch_size=1,
        detector_batch_size=1,
        pcutoff=0.1,
        customized_pose_checkpoint=pose_ckpt,
        customized_detector_checkpoint=detector_ckpt,
    )
    _save_results_to_csv_and_cleanup(results, video_path, dest_folder)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLabCut 3.0 无 GUI 关键点导出 CSV")
    parser.add_argument("--video", type=str, default="test/VAGotrdRsWk.mp4", help="输入视频路径")
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument("--pose", type=str, default=None, help="DLC3.0：本地姿态模型 checkpoint (.pt) 路径")
    parser.add_argument("--detector", type=str, default=None, help="DLC3.0：本地检测器 checkpoint (.pt) 路径")
    parser.add_argument("--hf-endpoint", type=str, default=None, help="HuggingFace 镜像，例如 https://hf-mirror.com")
    parser.add_argument("--local-only", action="store_true", help="仅离线：禁止联网下载，必须配合 --pose/--detector")
    args = parser.parse_args()

    video_path = args.video
    dest_folder = args.output

    # 设置 HuggingFace 镜像或离线模式（若提供）
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # 运行 DLC 3.0 推理（可传入本地权重以避免下载）
    _run_dlc30(video_path, dest_folder, args.pose, args.detector)


if __name__ == "__main__":
    main()