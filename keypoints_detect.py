'''
示例
python /root/autodl-tmp/AAH/keypoints_detect.py \
  --video /root/autodl-tmp/AAH/test/VAGotrdRsWk.mp4 \
  --output /root/autodl-tmp/AAH/output \
  --hf-endpoint https://hf-mirror.com
'''

import os
import argparse
from pathlib import Path
import glob
import uuid
from typing import List

try:
    import cv2  # 用于将多张图片拼接为视频（回退路径）
except Exception:  # 仅在需要回退时才会真正用到
    cv2 = None

# 禁用一切 GUI（在导入 deeplabcut 前设置）
os.environ["DLClight"] = "True"
os.environ.setdefault("MPLBACKEND", "Agg")


def _save_results_to_csv_and_cleanup(results: dict, video_path: str, dest_folder: str, save_vis: bool = False) -> None:
    """将推理结果 DataFrame 保存为 CSV，并清理中间文件（可选保留可视化视频）。

    /**
     * @param {dict} results - `video_inference_superanimal` 返回的 {video_path: pd.DataFrame} 映射
     * @param {str} video_path - 输入视频路径
     * @param {str} dest_folder - 输出目录
     * @param {bool} save_vis - 是否保留 DLC 生成的可视化视频（*_labeled*.mp4）
     * @returns {None}
     */
    """
    for _vid, df in results.items():
        stem = Path(video_path).stem
        csv_path = Path(dest_folder) / f"{stem}_keypoints_det.csv"
        df.to_csv(csv_path)

    stem = Path(video_path).stem
    patterns = []
    if not save_vis:
        patterns.append(f"{stem}_*_labeled*.mp4")
    patterns.extend([
        f"{stem}_*.h5",
        f"{stem}_*.json",
    ])
    for pat in patterns:
        for fpath in glob.glob(str(Path(dest_folder) / pat)):
            try:
                os.remove(fpath)
            except OSError:
                pass


def _save_image_results_to_csv(results: dict, dest_folder: str) -> None:
    """将图片推理结果分别保存为 CSV（每张图片一个 CSV）。

    /**
     * @param {dict} results - `image_inference_superanimal` 返回的 {image_path: pd.DataFrame} 映射
     * @param {str} dest_folder - 输出目录
     * @returns {None}
     */
    """
    for img_path, df in results.items():
        stem = Path(img_path).stem
        csv_path = Path(dest_folder) / f"{stem}_keypoints_det.csv"
        df.to_csv(csv_path)


def _write_per_frame_csv(df, image_paths: List[str], dest_folder: str) -> None:
    """将视频推理返回的单个 DataFrame 按帧拆分，逐张图片输出 CSV。

    /**
     * @param {pd.DataFrame} df - DLC 视频推理的 DataFrame（多帧）
     * @param {Array<string>} image_paths - 与帧顺序一致的图片路径数组
     * @param {str} dest_folder - 输出目录
     * @returns {None}
     */
    """
    num_frames = len(df)
    if num_frames != len(image_paths):
        # 若帧数与图片数不匹配，尽量按最小长度输出，避免崩溃
        limit = min(num_frames, len(image_paths))
    else:
        limit = num_frames
    for i in range(limit):
        img_path = image_paths[i]
        stem = Path(img_path).stem
        csv_path = Path(dest_folder) / f"{stem}_keypoints_det.csv"
        df.iloc[i : i + 1].to_csv(csv_path)


def _run_dlc30(
    video_path: str,
    dest_folder: str,
    pose_ckpt: str | None = None,
    detector_ckpt: str | None = None,
    save_vis: bool = False,
) -> None:
    """使用 DLC 3.0 的 SuperAnimal API 运行推理。

    /**
     * @param {str} video_path - 输入视频
     * @param {str} dest_folder - 输出目录
     * @param {str | None} pose_ckpt - 可选，自定义姿态模型 checkpoint 路径（.pt）
     * @param {str | None} detector_ckpt - 可选，自定义检测器 checkpoint 路径（.pt）
     * @param {bool} save_vis - 是否保存可视化视频
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
        plot_bboxes=save_vis,
        batch_size=1,
        detector_batch_size=1,
        pcutoff=0.1,
        customized_pose_checkpoint=pose_ckpt,
        customized_detector_checkpoint=detector_ckpt,
    )
    _save_results_to_csv_and_cleanup(results, video_path, dest_folder, save_vis)


def _run_dlc30_images(
    image_paths: list[str],
    dest_folder: str,
    pose_ckpt: str | None = None,
    detector_ckpt: str | None = None,
    save_vis: bool = False,
) -> None:
    """使用 DLC 3.0 的 SuperAnimal API 对多张图片运行推理。

    /**
     * @param {Array<string>} image_paths - 图片路径列表
     * @param {str} dest_folder - 输出目录
     * @param {str | None} pose_ckpt - 可选，自定义姿态模型 checkpoint 路径（.pt）
     * @param {str | None} detector_ckpt - 可选，自定义检测器 checkpoint 路径（.pt）
     * @param {bool} save_vis - 是否保存可视化视频（若图片 API 不支持，将回退到视频路径以生成可视化）
     * @returns {None}
     */
    """
    # 优先尝试图片推理 API；若不存在则回退为“拼接临时视频再视频推理”。
    try:
        from deeplabcut.modelzoo.image_inference import image_inference_superanimal
    except Exception:
        image_inference_superanimal = None

    os.makedirs(dest_folder, exist_ok=True)

    # 若需要保存可视化视频，则直接使用回退的视频路径，确保能生成 labeled 视频
    if image_inference_superanimal is not None and not save_vis:
        results = image_inference_superanimal(
            images=image_paths,
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
        _save_image_results_to_csv(results, dest_folder)
        return

    # 回退实现：将图片序列写入临时视频，再用视频推理；随后按帧拆分结果到逐图 CSV
    if cv2 is None:
        raise SystemExit(
            "OpenCV 未安装，且 DLC3.0 图片推理 API 不可用。请安装 opencv-python 或升级 DLC3.0 以支持图片推理。"
        )

    # 读取首图，确定视频分辨率
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        raise SystemExit(f"无法读取图片：{image_paths[0]}")
    height, width = first_img.shape[:2]

    tmp_stem = f"_dlcimg_{uuid.uuid4().hex[:8]}"
    tmp_video = str(Path(dest_folder) / f"{tmp_stem}.mp4")

    # 写入视频（保持所有图片 resize 到首图大小）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, 1.0, (width, height))
    if not writer.isOpened():
        raise SystemExit("无法创建临时视频文件，请检查 OpenCV 的编解码器支持。")
    try:
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                raise SystemExit(f"无法读取图片：{img_path}")
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))
            writer.write(img)
    finally:
        writer.release()

    # 调用视频推理，获取 DataFrame 并按帧输出 CSV
    try:
        from deeplabcut.modelzoo.video_inference import video_inference_superanimal
    except Exception as exc:
        raise SystemExit(
            "未检测到可用的 DeepLabCut 3.0 视频推理 API。原始错误：" + str(exc)
        )

    results = video_inference_superanimal(
        videos=[tmp_video],
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

    # results: {video_path: df}
    df = results.get(tmp_video)
    if df is None:
        raise SystemExit("视频推理未返回结果。")
    _write_per_frame_csv(df, image_paths, dest_folder)

    # 清理 DLC 产生的中间文件和临时视频；可选保留可视化视频
    stem = Path(tmp_video).stem
    patterns = [
        f"{stem}_*.h5",
        f"{stem}_*.json",
        f"{stem}_*.csv",
    ]
    # 若不保存可视化视频，则一并删除 labeled 视频
    if not save_vis:
        patterns.insert(0, f"{stem}_*_labeled*.mp4")
    for pat in patterns:
        for fpath in glob.glob(str(Path(dest_folder) / pat)):
            try:
                os.remove(fpath)
            except OSError:
                pass
    # 如果需要保留可视化视频，尝试将临时命名的 labeled 视频重命名为更友好的输出名
    if save_vis:
        labeled_candidates = list(Path(dest_folder).glob(f"{stem}_*_labeled*.mp4"))
        if labeled_candidates:
            target_name = f"{Path(image_paths[0]).parent.name}_labeled.mp4"
            try:
                labeled_candidates[0].rename(Path(dest_folder) / target_name)
            except OSError:
                pass
    # 删除临时视频
    try:
        os.remove(tmp_video)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLabCut 3.0 无 GUI 关键点导出 CSV")
    parser.add_argument(
        "--video",
        type=str,
        default="test/VAGotrdRsWk.mp4",
        help="输入路径：视频文件、单张图片或包含图片的文件夹",
    )
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument("--pose", type=str, default=None, help="DLC3.0：本地姿态模型 checkpoint (.pt) 路径")
    parser.add_argument("--detector", type=str, default=None, help="DLC3.0：本地检测器 checkpoint (.pt) 路径")
    parser.add_argument("--hf-endpoint", type=str, default=None, help="HuggingFace 镜像，例如 https://hf-mirror.com")
    parser.add_argument("--local-only", action="store_true", help="仅离线：禁止联网下载，必须配合 --pose/--detector")
    parser.add_argument("--save-vis", action="store_true", help="保存关键点检测的可视化视频")
    args = parser.parse_args()

    video_path = args.video
    dest_folder = args.output

    # 设置 HuggingFace 镜像或离线模式（若提供）
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # 判断输入类型：文件夹/单张图片/视频
    input_path = Path(video_path)
    if input_path.is_dir():
        # 收集文件夹下的图片文件（非递归，可按需改为递归）
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
        image_paths = []
        for ext in exts:
            image_paths.extend([str(p) for p in input_path.glob(ext)])
        if not image_paths:
            raise SystemExit(f"在目录中未找到图片文件：{input_path}")
        _run_dlc30_images(image_paths, dest_folder, args.pose, args.detector, args.save_vis)
        return

    # 文件：根据扩展名区分图片或视频
    lower_name = input_path.name.lower()
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    if lower_name.endswith(image_exts):
        _run_dlc30_images([str(input_path)], dest_folder, args.pose, args.detector, args.save_vis)
    else:
        # 默认按视频处理
        _run_dlc30(str(input_path), dest_folder, args.pose, args.detector, args.save_vis)


if __name__ == "__main__":
    main()