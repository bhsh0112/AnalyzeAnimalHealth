import os
import sys
import argparse
from pathlib import Path
import glob

# 禁用一切 GUI（在导入 deeplabcut 前设置）
os.environ["DLClight"] = "True"
os.environ.setdefault("MPLBACKEND", "Agg")


def _save_results_to_csv_and_cleanup(results: dict, video_path: str, dest_folder: str) -> None:
    """将推理结果 DataFrame 保存为 CSV，并删除多余的可视化视频和中间文件。

    Args:
        results: video_inference_superanimal 返回的 {video_path: pd.DataFrame} 映射
        video_path: 输入视频路径
        dest_folder: 输出目录
    """
    for _vid, df in results.items():
        stem = Path(video_path).stem
        csv_path = Path(dest_folder) / f"{stem}_predictions.csv"
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
) -> bool:
    """尝试使用 DLC 3.0 的 SuperAnimal API 运行推理。返回是否成功。

    Args:
        video_path: 输入视频
        dest_folder: 输出目录
        pose_ckpt: 可选，自定义姿态模型 checkpoint 路径（.pt）
        detector_ckpt: 可选，自定义检测器 checkpoint 路径（.pt）
    """
    try:
        from deeplabcut.modelzoo.video_inference import video_inference_superanimal
    except Exception:
        return False

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
    return True


def _run_dlc23(config_path: str, video_path: str, dest_folder: str) -> None:
    """使用 DLC 2.3 的 analyze_videos，仅输出 CSV。"""
    import deeplabcut

    os.makedirs(dest_folder, exist_ok=True)
    deeplabcut.analyze_videos(
        config_path,
        [video_path],
        videotype="mp4",
        destfolder=dest_folder,
        save_as_csv=True,
        save_as_h5=True,
    )
    # DLC2.3 会同时生成 h5；此处仅保留 CSV
    stem = Path(video_path).stem
    for fpath in glob.glob(str(Path(dest_folder) / f"{stem}_*.h5")):
        try:
            os.remove(fpath)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLabCut 无 GUI 关键点导出 CSV")
    parser.add_argument("--video", type=str, default="test/VAGotrdRsWk.mp4", help="输入视频路径")
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="DLC 2.3 模式所需的项目 config.yaml 路径；当 3.0 API 不可用时必填",
    )
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

    # 先尝试 DLC 3.0 模式（可传入本地权重以避免下载）
    ran = _run_dlc30(video_path, dest_folder, args.pose, args.detector)
    if ran:
        return

    # 回退到 DLC 2.3 模式
    if not args.config:
        print(
            "未检测到 DLC 3.0 API，且未提供 --config。请提供 DLC 2.3 项目的 config.yaml 路径以继续。",
            file=sys.stderr,
        )
        sys.exit(2)
    _run_dlc23(args.config, video_path, dest_folder)


if __name__ == "__main__":
    main()