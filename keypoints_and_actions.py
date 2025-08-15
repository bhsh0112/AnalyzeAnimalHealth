'''
集成关键点检测和姿态判断的完整解决方案

示例使用：
python /root/autodl-tmp/AAH/keypoints_and_actions.py \
  --video /root/autodl-tmp/AAH/test/VAGotrdRsWk.mp4 \
  --output /root/autodl-tmp/AAH/output \
  --hf-endpoint https://hf-mirror.com \
  --save-vis

功能特性：
1. 逐帧关键点检测（基于 DLC 3.0）
2. 实时姿态分类（upright / upright_head_down / lying）
3. 增强可视化（在关键点基础上叠加姿态标签）
4. 输出关键点CSV和姿态CSV
'''

import os
import argparse
import math
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import cv2  # 用于视频处理和可视化
except Exception:
    cv2 = None

# 禁用一切 GUI（在导入 deeplabcut 前设置）
os.environ["DLClight"] = "True"
os.environ.setdefault("MPLBACKEND", "Agg")


@dataclass
class ActionResult:
    """单帧动作分析结果"""
    frame: int
    animal: str
    action: str
    speed_px: float
    body_length_px: float
    speed_norm_body: float
    center_x: float
    center_y: float


class RealTimeActionClassifier:
    """实时姿态分类器，处理逐帧关键点数据"""

    def __init__(
        self,
        min_likelihood: float = 0.5,
        smoothing_window: int = 5,
        torso_upright_angle_deg: float = 40.0,
        torso_lying_angle_deg: float = 65.0,
        head_down_drop_frac: float = 0.15,
        lying_gap_frac: float = 0.25,
    ):
        """
        初始化实时姿态分类器

        /**
         * @param {number} min_likelihood - 最小置信度阈值
         * @param {number} smoothing_window - 历史中心平滑窗口，用于稳定可视化位置
         * @param {number} torso_upright_angle_deg - 躯干相对垂直方向的角度阈值（小于该值视为直立）
         * @param {number} torso_lying_angle_deg - 躯干相对垂直方向的角度阈值（大于该值视为卧倒）
         * @param {number} head_down_drop_frac - 头部相对颈部的下探距离阈值（相对于体长，超过视为低头）
         */
        """
        self.min_likelihood = min_likelihood
        self.smoothing_window = smoothing_window
        self.torso_upright_angle_deg = torso_upright_angle_deg
        self.torso_lying_angle_deg = torso_lying_angle_deg
        self.head_down_drop_frac = head_down_drop_frac
        self.lying_gap_frac = lying_gap_frac

        # 存储历史数据（位置平滑用）
        self.history: Dict[str, Dict[str, List[float]]] = {}
        self.body_lengths: Dict[str, float] = {}
    
    def _safe_float(self, x) -> float:
        """安全转换为浮点数"""
        try:
            return float(x)
        except Exception:
            return float("nan")
    
    def _extract_keypoint(self, frame_data: pd.Series, individual: str, bodypart: str) -> Optional[Tuple[float, float, float]]:
        """从单帧数据中提取关键点坐标和置信度
        
        /**
         * @param {pd.Series} frame_data - 单帧的关键点数据
         * @param {string} individual - 个体ID
         * @param {string} bodypart - 关键点名称
         * @returns {[number, number, number] | None} [x, y, likelihood] 或 None
         */
        """
        try:
            # DLC 多级列格式：(scorer, individual, bodypart, coord)
            x_cols = [col for col in frame_data.index if col[1] == individual and col[2] == bodypart and col[3] == 'x']
            y_cols = [col for col in frame_data.index if col[1] == individual and col[2] == bodypart and col[3] == 'y']
            l_cols = [col for col in frame_data.index if col[1] == individual and col[2] == bodypart and col[3] == 'likelihood']
            
            if not (x_cols and y_cols and l_cols):
                return None
            
            x = self._safe_float(frame_data[x_cols[0]])
            y = self._safe_float(frame_data[y_cols[0]])
            likelihood = self._safe_float(frame_data[l_cols[0]])
            
            if likelihood < self.min_likelihood:
                return None
            
            return x, y, likelihood
        except Exception:
            return None
    
    def _estimate_torso_center(self, frame_data: pd.Series, individual: str) -> Optional[Tuple[float, float]]:
        """估计躯干中心点
        
        /**
         * @param {pd.Series} frame_data - 单帧关键点数据
         * @param {string} individual - 个体ID
         * @returns {[number, number] | None} [x, y] 或 None
         */
        """
        # 优先使用 back_middle
        center = self._extract_keypoint(frame_data, individual, "back_middle")
        if center:
            return center[0], center[1]
        
        # 使用 back_base 和 back_end 的中点
        back_base = self._extract_keypoint(frame_data, individual, "back_base")
        back_end = self._extract_keypoint(frame_data, individual, "back_end")
        if back_base and back_end:
            return (back_base[0] + back_end[0]) / 2, (back_base[1] + back_end[1]) / 2
        
        # 使用 neck_base 和 tail_base 的中点
        neck_base = self._extract_keypoint(frame_data, individual, "neck_base")
        tail_base = self._extract_keypoint(frame_data, individual, "tail_base")
        if neck_base and tail_base:
            return (neck_base[0] + tail_base[0]) / 2, (neck_base[1] + tail_base[1]) / 2
        
        return None
    
    def _estimate_body_length(self, frame_data: pd.Series, individual: str) -> float:
        """估计身体长度（像素）
        
        /**
         * @param {pd.Series} frame_data - 单帧关键点数据
         * @param {string} individual - 个体ID
         * @returns {number} 身体长度（像素）
         */
        """
        # 优先使用 neck_base 到 tail_base 的距离
        neck_base = self._extract_keypoint(frame_data, individual, "neck_base")
        tail_base = self._extract_keypoint(frame_data, individual, "tail_base")
        if neck_base and tail_base:
            dx = neck_base[0] - tail_base[0]
            dy = neck_base[1] - tail_base[1]
            return math.sqrt(dx*dx + dy*dy)
        
        # 使用 back_base 到 back_end 的距离
        back_base = self._extract_keypoint(frame_data, individual, "back_base")
        back_end = self._extract_keypoint(frame_data, individual, "back_end")
        if back_base and back_end:
            dx = back_base[0] - back_end[0]
            dy = back_base[1] - back_end[1]
            return math.sqrt(dx*dx + dy*dy)
        
        # 使用历史平均值或默认值
        if individual in self.body_lengths:
            return self.body_lengths[individual]
        
        return 100.0  # 默认值

    def _estimate_head_point(self, frame_data: pd.Series, individual: str) -> Optional[Tuple[float, float]]:
        """估计头部关键点位置（若存在）。

        /**
         * @param {pd.Series} frame_data - 单帧关键点数据
         * @param {string} individual - 个体ID
         * @returns {[number, number] | None} [x, y] 或 None
         */
        """
        head_candidates = [
            "nose",
            "snout",
            "head",
            "head_top",
            "head_front",
            "mouth",
        ]
        for part in head_candidates:
            pt = self._extract_keypoint(frame_data, individual, part)
            if pt:
                return pt[0], pt[1]
        return None

    def _max_valid_y(self, frame_data: pd.Series, individual: str) -> Optional[float]:
        """获取该个体所有可用关键点中最大的 y（图像坐标向下为正）。

        /**
         * @param {pd.Series} frame_data - 单帧关键点数据
         * @param {string} individual - 个体ID
         * @returns {number | None} 最大 y 值
         */
        """
        max_y = None
        try:
            # MultiIndex: (scorer, individual, bodypart, coord)
            for col in frame_data.index:
                if len(col) < 4:
                    continue
                if col[1] != individual:
                    continue
                if col[3] != 'y':
                    continue
                # 找对应的 likelihood 列
                l_col = (col[0], col[1], col[2], 'likelihood')
                likelihood = frame_data.get(l_col, 1.0)
                if self._safe_float(likelihood) < self.min_likelihood:
                    continue
                y_val = self._safe_float(frame_data[col])
                if not math.isfinite(y_val):
                    continue
                if max_y is None or y_val > max_y:
                    max_y = y_val
        except Exception:
            return None
        return max_y

    def _compute_torso_angle_to_vertical_deg(
        self, frame_data: pd.Series, individual: str
    ) -> Optional[float]:
        """计算躯干轴线相对垂直方向的夹角（度）。0 表示完全垂直，90 表示完全水平。

        /**
         * @param {pd.Series} frame_data - 单帧关键点数据
         * @param {string} individual - 个体ID
         * @returns {number | None} 角度（度）
         */
        """
        # 使用 neck_base ←→ tail_base 或 back_base ←→ back_end 作为躯干轴
        a = self._extract_keypoint(frame_data, individual, "neck_base")
        b = self._extract_keypoint(frame_data, individual, "tail_base")
        if not (a and b):
            a = self._extract_keypoint(frame_data, individual, "back_base")
            b = self._extract_keypoint(frame_data, individual, "back_end")
        if not (a and b):
            return None
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        angle_h = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9))  # 与水平的角度（0~90）
        angle_to_vertical = abs(90.0 - angle_h)
        return angle_to_vertical

    def _is_head_down(
        self, frame_data: pd.Series, individual: str, body_length: float
    ) -> Optional[bool]:
        """判断是否低头。首选 head 相对 neck 的下探；若无 head，则用 neck 相对 back_middle 的下探近似。

        /**
         * @param {pd.Series} frame_data - 单帧关键点数据
         * @param {string} individual - 个体ID
         * @param {number} body_length - 身体长度（像素）
         * @returns {boolean | None} 是否低头；无法判断返回 None
         */
        """
        neck = self._extract_keypoint(frame_data, individual, "neck_base")
        if neck is None:
            return None
        head = self._estimate_head_point(frame_data, individual)
        if head is not None:
            drop = head[1] - neck[1]  # y 向下为正
            return drop > self.head_down_drop_frac * max(body_length, 1e-6)
        # 回退：使用 back_middle 作为参考（neck 明显低于背部中心视为低头）
        back_mid = self._extract_keypoint(frame_data, individual, "back_middle")
        if back_mid:
            drop = neck[1] - back_mid[1]
            return drop > self.head_down_drop_frac * max(body_length, 1e-6)
        return None
    
    def _rolling_mean(self, values: List[float], window: int) -> float:
        """计算滑动平均"""
        if not values:
            return 0.0
        n = min(len(values), window)
        return sum(values[-n:]) / n
    
    def process_frame(self, frame_data: pd.Series, frame_num: int) -> List[ActionResult]:
        """处理单帧数据，返回所有个体的动作结果
        
        /**
         * @param {pd.Series} frame_data - 单帧关键点数据
         * @param {number} frame_num - 帧号
         * @returns {Array<ActionResult>} 所有个体的动作分析结果
         */
        """
        results = []
        
        # 获取所有个体
        individuals = set()
        for col in frame_data.index:
            if len(col) >= 2:
                individuals.add(col[1])
        
        for individual in individuals:
            # 估计躯干中心和身体长度
            center = self._estimate_torso_center(frame_data, individual)
            if center is None:
                continue
            
            body_length = self._estimate_body_length(frame_data, individual)
            
            # 初始化历史数据
            if individual not in self.history:
                self.history[individual] = {
                    'center_x': [],
                    'center_y': [],
                    'speed': []
                }
                self.body_lengths[individual] = body_length
            else:
                # 更新身体长度的滑动平均
                self.body_lengths[individual] = 0.9 * self.body_lengths[individual] + 0.1 * body_length
            
            # 速度对姿态分类非必需，仅用于保持与现有输出字段兼容
            speed_px = 0.0

            # 更新历史数据
            self.history[individual]['center_x'].append(center[0])
            self.history[individual]['center_y'].append(center[1])
            self.history[individual]['speed'].append(speed_px)
            
            # 限制历史数据长度
            max_history = self.smoothing_window * 2
            for key in self.history[individual]:
                if len(self.history[individual][key]) > max_history:
                    self.history[individual][key] = self.history[individual][key][-max_history:]
            
            # 计算平滑速度（保持字段填充，不参与分类）
            smooth_speed = self._rolling_mean(self.history[individual]['speed'], self.smoothing_window)
            speed_norm = smooth_speed / self.body_lengths[individual] if self.body_lengths[individual] > 1e-6 else 0.0

            # 姿态分类：优先使用“垂直间隙”规则区分 lying/upright，其次再判定低头
            posture = "upright"
            max_y = self._max_valid_y(frame_data, individual)
            if max_y is not None and math.isfinite(center[1]) and self.body_lengths[individual] > 1e-6:
                vertical_gap = max_y - center[1]
                if vertical_gap <= self.lying_gap_frac * self.body_lengths[individual]:
                    posture = "lying"
                else:
                    head_down = self._is_head_down(frame_data, individual, self.body_lengths[individual])
                    posture = "upright_head_down" if head_down else "upright"
            else:
                # 回退：使用躯干角度作为辅助（不激进判 lying）
                torso_angle_to_vertical = self._compute_torso_angle_to_vertical_deg(frame_data, individual)
                if torso_angle_to_vertical is not None and torso_angle_to_vertical >= self.torso_lying_angle_deg:
                    posture = "lying"
                else:
                    head_down = self._is_head_down(frame_data, individual, self.body_lengths[individual])
                    posture = "upright_head_down" if head_down else "upright"
            
            results.append(ActionResult(
                frame=frame_num,
                animal=individual,
                action=posture,
                speed_px=smooth_speed,
                body_length_px=self.body_lengths[individual],
                speed_norm_body=speed_norm,
                center_x=center[0],
                center_y=center[1]
            ))
        
        return results


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


def _enhance_visualization_with_actions(video_path: str, dest_folder: str, action_results: List[ActionResult]) -> None:
    """在可视化视频上添加动作标签
    
    /**
     * @param {string} video_path - 原始视频路径
     * @param {string} dest_folder - 输出目录
     * @param {Array<ActionResult>} action_results - 动作分析结果
     * @returns {None}
     */
    """
    if cv2 is None:
        print("OpenCV 未安装，跳过可视化增强")
        return
    
    stem = Path(video_path).stem
    # 查找 DLC 生成的可视化视频
    labeled_videos = glob.glob(str(Path(dest_folder) / f"{stem}_*_labeled*.mp4"))
    if not labeled_videos:
        print("未找到 DLC 可视化视频，跳过动作标签添加")
        return
    
    labeled_video = labeled_videos[0]
    output_video = str(Path(dest_folder) / f"{stem}_with_actions.mp4")
    
    # 按帧组织动作结果
    actions_by_frame: Dict[int, Dict[str, ActionResult]] = {}
    for result in action_results:
        if result.frame not in actions_by_frame:
            actions_by_frame[result.frame] = {}
        actions_by_frame[result.frame][result.animal] = result
    
    # 读取原视频并添加动作标签
    cap = cv2.VideoCapture(labeled_video)
    if not cap.isOpened():
        print(f"无法打开视频：{labeled_video}")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    action_colors = {
        'upright': (0, 255, 0),            # 绿色
        'upright_head_down': (255, 255, 0),# 黄色
        'lying': (0, 0, 255)               # 红色
    }
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 添加动作标签
            if frame_idx in actions_by_frame:
                y_offset = 30
                for animal, result in actions_by_frame[frame_idx].items():
                    # 在动物中心位置附近显示动作标签
                    x = int(result.center_x)
                    y = int(result.center_y - 20)  # 稍微上移避免遮挡关键点
                    
                    color = action_colors.get(result.action, (255, 255, 255))
                    text = f"{animal}: {result.action}"
                    
                    # 添加文本背景
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x-5, y-text_height-5), (x+text_width+5, y+5), (0, 0, 0), -1)
                    
                    # 添加文本
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 在左上角显示速度信息
                    speed_text = f"{animal} speed: {result.speed_norm_body:.3f}"
                    cv2.putText(frame, speed_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 25
            
            out.write(frame)
            frame_idx += 1
            
    finally:
        cap.release()
        out.release()
    
    print(f"增强可视化视频已保存到：{output_video}")


def _run_dlc30_with_actions(
    video_path: str,
    dest_folder: str,
    pose_ckpt: str | None = None,
    detector_ckpt: str | None = None,
    save_vis: bool = False,
    action_params: dict = None,
) -> None:
    """使用 DLC 3.0 进行关键点检测并实时进行动作分析。

    /**
     * @param {str} video_path - 输入视频
     * @param {str} dest_folder - 输出目录
     * @param {str | None} pose_ckpt - 可选，自定义姿态模型 checkpoint 路径（.pt）
     * @param {str | None} detector_ckpt - 可选，自定义检测器 checkpoint 路径（.pt）
     * @param {bool} save_vis - 是否保存可视化视频
     * @param {dict | None} action_params - 动作分类参数
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
    
    # 运行 DLC 关键点检测
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
    
    # 保存关键点结果
    _save_results_to_csv_and_cleanup(results, video_path, dest_folder, save_vis)
    
    # 实时动作分析
    action_params = action_params or {}
    classifier = RealTimeActionClassifier(**action_params)
    
    # 获取关键点数据
    keypoints_df = list(results.values())[0]  # 假设只有一个视频
    
    all_action_results = []
    for frame_idx in keypoints_df.index:
        frame_data = keypoints_df.loc[frame_idx]
        frame_results = classifier.process_frame(frame_data, frame_idx)
        all_action_results.extend(frame_results)
    
    # 保存动作分析结果
    if all_action_results:
        # 获取视频 FPS
        fps = None
        if cv2:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or None
                cap.release()
        
        # 创建姿态结果 DataFrame
        action_data = []
        for result in all_action_results:
            row = {
                'frame': result.frame,
                'animal': result.animal,
                'posture': result.action,
                'speed_px': result.speed_px,
                'body_length_px': result.body_length_px,
                'speed_norm_body': result.speed_norm_body
            }
            if fps and fps > 0:
                row['time_s'] = result.frame / fps
            action_data.append(row)
        
        action_df = pd.DataFrame(action_data)
        
        # 调整列顺序
        if fps and fps > 0:
            action_df = action_df[['frame', 'time_s', 'animal', 'posture', 'speed_px', 'body_length_px', 'speed_norm_body']]
        else:
            action_df = action_df[['frame', 'animal', 'posture', 'speed_px', 'body_length_px', 'speed_norm_body']]
        
        # 保存姿态分析结果
        stem = Path(video_path).stem
        action_csv_path = Path(dest_folder) / f"{stem}_postures.csv"
        action_df.to_csv(action_csv_path, index=False)
        print(f"姿态分析结果已保存到：{action_csv_path}")
        
        # 如果保存可视化，则增强可视化结果
        if save_vis:
            _enhance_visualization_with_actions(video_path, dest_folder, all_action_results)


def _images_to_video(image_paths: list[str], dest_folder: str, fps: float = 20) -> str:
    """将图片序列按首图分辨率拼接成临时视频并返回视频路径。

    /**
     * @param {Array<string>} image_paths - 需要按顺序写入的视频帧图片路径
     * @param {str} dest_folder - 输出目录（临时视频将写入此目录）
     * @param {number} fps - 临时视频帧率
     * @returns {str} 临时视频文件的绝对路径
     */
    """
    if cv2 is None:
        raise SystemExit(
            "OpenCV 未安装，无法将图片拼接为视频。请先安装 opencv-python。"
        )

    os.makedirs(dest_folder, exist_ok=True)

    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        raise SystemExit(f"无法读取图片：{image_paths[0]}")
    height, width = first_img.shape[:2]

    # 使用所在目录名作为视频文件名，使后续输出与视频处理命名规则保持一致
    folder_name = Path(image_paths[0]).parent.name
    tmp_video = str(Path(dest_folder) / f"{folder_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, float(fps), (width, height))
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

    return tmp_video


def main() -> None:
    parser = argparse.ArgumentParser(description="集成关键点检测和姿态分类的完整解决方案")
    parser.add_argument(
        "--video",
        type=str,
        default="test/VAGotrdRsWk.mp4",
        help="输入路径：视频文件或包含图片的文件夹（不支持单张图片）",
    )
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument("--pose", type=str, default=None, help="DLC3.0：本地姿态模型 checkpoint (.pt) 路径")
    parser.add_argument("--detector", type=str, default=None, help="DLC3.0：本地检测器 checkpoint (.pt) 路径")
    parser.add_argument("--hf-endpoint", type=str, default=None, help="HuggingFace 镜像，例如 https://hf-mirror.com")
    parser.add_argument("--local-only", action="store_true", help="仅离线：禁止联网下载，必须配合 --pose/--detector")
    parser.add_argument("--save-vis", action="store_true", help="保存关键点检测的可视化视频（含姿态标签）")
    
    # 姿态分类参数
    parser.add_argument("--min-likelihood", type=float, default=0.5, help="最小置信度阈值")
    parser.add_argument("--smooth-win", type=int, default=5, help="位置平滑窗口（帧）")
    parser.add_argument("--torso-upright-deg", type=float, default=40.0, help="upright 阈值：躯干相对垂直角度 ≤ 此值判为 upright/upright_head_down")
    parser.add_argument("--torso-lying-deg", type=float, default=65.0, help="lying 阈值：躯干相对垂直角度 ≥ 此值判为 lying")
    parser.add_argument("--head-down-frac", type=float, default=0.15, help="upright_head_down 阈值：头部相对颈部下探距离阈值（体长比例）")
    parser.add_argument("--lying-gap-frac", type=float, default=0.25, help="lying 判定阈值：躯干中心到所有关键点最大 y 的垂直间隙占体长比例上限")
    
    args = parser.parse_args()

    video_path = args.video
    dest_folder = args.output

    # 设置 HuggingFace 镜像或离线模式（若提供）
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # 姿态分类参数
    action_params = {
        'min_likelihood': args.min_likelihood,
        'smoothing_window': args.smooth_win,
        'torso_upright_angle_deg': args.torso_upright_deg,
        'torso_lying_angle_deg': args.torso_lying_deg,
        'head_down_drop_frac': args.head_down_frac,
        'lying_gap_frac': args.lying_gap_frac,
    }

    # 判断输入类型：文件夹/视频（不支持单张图片）
    input_path = Path(video_path)
    if input_path.is_dir():
        # 收集文件夹下的图片文件（非递归，可按需改为递归）
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
        image_paths = []
        for ext in exts:
            image_paths.extend([str(p) for p in input_path.glob(ext)])
        if not image_paths:
            raise SystemExit(f"在目录中未找到图片文件：{input_path}")
        # 排序以保证帧顺序可复现
        image_paths = sorted(image_paths)
        # 先拼接为视频，再按视频路径执行与视频一致的流程与输出
        tmp_video = _images_to_video(image_paths, dest_folder, fps=20)
        try:
            _run_dlc30_with_actions(tmp_video, dest_folder, args.pose, args.detector, args.save_vis, action_params)
        finally:
            try:
                os.remove(tmp_video)
            except OSError:
                pass
        return

    # 文件：根据扩展名区分图片或视频
    lower_name = input_path.name.lower()
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    if lower_name.endswith(image_exts):
        raise SystemExit("当前不支持单张图片输入。请将图片放入文件夹后作为目录输入。")
    else:
        # 默认按视频处理
        _run_dlc30_with_actions(str(input_path), dest_folder, args.pose, args.detector, args.save_vis, action_params)


if __name__ == "__main__":
    main()
