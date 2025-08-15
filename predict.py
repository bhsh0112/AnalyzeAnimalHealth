import os
import glob
import math
import json
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
import deeplabcut
from deeplabcut.modelzoo.api import SpatiotemporalAdaptation


# /**
#  * 基于 DeepLabCut 的关键点检测结果，提供骨架渲染、关节角度计算、
#  * 置信度过滤、时序平滑以及简单的姿态分类（站立/低头/抬头/未知）。
#  * 参考与依赖: DeepLabCut 官方实现与模型。
#  * @see https://github.com/DeepLabCut/DeepLabCut?tab=readme-ov-file
#  */


# 关键点命名（与项目训练时的 bodyparts 名称保持一致或可被 CSV 中自动发现）
DEFAULT_KEYPOINT_NAMES: List[str] = [
    'nostril',  # 鼻孔
    'eye',      # 眼睛（可为任一侧）
    'ear',      # 耳朵（可为任一侧）
    'withers',  # 鬐甲（肩部最高点）
    'elbow',    # 肘部（前肢）
    'knee',     # 膝盖（前膝）
    'coronet',  # 蹄冠
    'hock',     # 飞节（后肢）
    'tail_base' # 尾根
]


# 骨架连线定义（可按需调整）
SKELETON_EDGES: List[Tuple[str, str]] = [
    ('nostril', 'eye'),
    ('eye', 'ear'),
    ('withers', 'elbow'),
    ('elbow', 'knee'),
    ('knee', 'coronet'),
    ('tail_base', 'hock'),
    ('hock', 'coronet'),
    ('withers', 'tail_base'),
]


# 颜色定义（BGR）
COLOR_POINT = (0, 215, 255)  # 金色
COLOR_EDGE = (0, 178, 255)   # 橙色
COLOR_TEXT = (0, 255, 0)     # 绿色
COLOR_WARN = (0, 0, 255)     # 红色


# /**
#  * 计算角度：以 b 为顶点，向量 ba 与 bc 的夹角（单位：度）。
#  * @param a Tuple[float, float] 点 A (x, y)
#  * @param b Tuple[float, float] 点 B (x, y) 顶点
#  * @param c Tuple[float, float] 点 C (x, y)
#  * @returns float 角度（0-180）
#  */
def compute_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    dot = bax * bcx + bay * bcy
    norm_ba = math.hypot(bax, bay)
    norm_bc = math.hypot(bcx, bcy)
    if norm_ba == 0 or norm_bc == 0:
        return float('nan')
    cos_val = max(-1.0, min(1.0, dot / (norm_ba * norm_bc)))
    return math.degrees(math.acos(cos_val))


# /**
#  * 指数滑动平均（EMA）平滑序列
#  * @param series np.ndarray 形如 (T,) 的序列
#  * @param alpha float 平滑系数，取值(0,1]，越大越信新值
#  * @returns np.ndarray 平滑后的序列
#  */
def ema_smooth(series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    if series.size == 0:
        return series
    out = np.array(series, dtype=float)
    prev = out[0]
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = prev
        else:
            prev = alpha * out[i] + (1 - alpha) * prev
            out[i] = prev
    return out


# /**
#  * 从 DeepLabCut 的 CSV 结果中读取关键点序列，并进行可选平滑与阈值过滤。
#  * @param csv_path str DeepLabCut analyze_videos 导出的 CSV 路径
#  * @param likelihood_threshold float 置信度阈值，低于该值的点将置为 NaN
#  * @param smooth_alpha float EMA 平滑系数，<=0 则不平滑
#  * @returns Tuple[pd.DataFrame, List[str], str] 返回处理后的 DataFrame、bodyparts 列表、scorer 名称
#  */
def load_and_prepare_dlc_csv(
    csv_path: str,
    likelihood_threshold: float = 0.5,
    smooth_alpha: float = 0.3
) -> Tuple[pd.DataFrame, List[str], str]:
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    scorers = df.columns.get_level_values(0).unique().tolist()
    scorer = scorers[0]
    bodyparts = df.columns.get_level_values(1).unique().tolist()

    # 置信度过滤与 EMA 平滑
    for bp in bodyparts:
        x_col = (scorer, bp, 'x')
        y_col = (scorer, bp, 'y')
        l_col = (scorer, bp, 'likelihood')
        if l_col in df.columns:
            low_conf_mask = df[l_col].values < likelihood_threshold
            df.loc[low_conf_mask, x_col] = np.nan
            df.loc[low_conf_mask, y_col] = np.nan
        if smooth_alpha and smooth_alpha > 0:
            df[x_col] = ema_smooth(df[x_col].values.astype(float), alpha=smooth_alpha)
            df[y_col] = ema_smooth(df[y_col].values.astype(float), alpha=smooth_alpha)
    return df, bodyparts, scorer


# /**
#  * 将一帧的关键点取出为 {name: (x, y, l)} 字典
#  * @param df pd.DataFrame 带多重列索引的 DLC 结果
#  * @param frame_idx int 帧下标
#  * @param bodyparts List[str] 关键点名称列表
#  * @param scorer str DLC scorer 名
#  * @returns Dict[str, Tuple[float, float, float]]
#  */
def extract_points_from_df(
    df: pd.DataFrame,
    frame_idx: int,
    bodyparts: List[str],
    scorer: str
) -> Dict[str, Tuple[float, float, float]]:
    pts: Dict[str, Tuple[float, float, float]] = {}
    for bp in bodyparts:
        x = df[(scorer, bp, 'x')].values[frame_idx] if (scorer, bp, 'x') in df.columns else float('nan')
        y = df[(scorer, bp, 'y')].values[frame_idx] if (scorer, bp, 'y') in df.columns else float('nan')
        l = df[(scorer, bp, 'likelihood')].values[frame_idx] if (scorer, bp, 'likelihood') in df.columns else float('nan')
        pts[bp] = (float(x), float(y), float(l))
    return pts


# /**
#  * 计算关节角度集合：肘部、膝盖、飞节、背线角度
#  * @param points Dict[str, Tuple[float,float,float]] 单帧关键点
#  * @returns Dict[str, float] 各角度（度）
#  */
def compute_joint_angles(points: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
    def p(name: str) -> Optional[Tuple[float, float]]:
        v = points.get(name, (float('nan'), float('nan'), float('nan')))
        return (v[0], v[1]) if not (math.isnan(v[0]) or math.isnan(v[1])) else None

    withers = p('withers')
    elbow = p('elbow')
    knee = p('knee')
    coronet = p('coronet')
    hock = p('hock')
    tail_base = p('tail_base')

    angles: Dict[str, float] = {}
    if withers and elbow and knee:
        angles['elbow_angle'] = compute_angle(withers, elbow, knee)
    else:
        angles['elbow_angle'] = float('nan')
    if elbow and knee and coronet:
        angles['knee_angle'] = compute_angle(elbow, knee, coronet)
    else:
        angles['knee_angle'] = float('nan')
    if tail_base and hock and coronet:
        angles['hock_angle'] = compute_angle(tail_base, hock, coronet)
    else:
        angles['hock_angle'] = float('nan')
    if withers and tail_base:
        dx = tail_base[0] - withers[0]
        dy = tail_base[1] - withers[1]
        angles['back_angle'] = math.degrees(math.atan2(dy, dx))
    else:
        angles['back_angle'] = float('nan')
    return angles


# /**
#  * 简单姿态分类（启发式）：
#  * - standing: 四肢角度位于合理范围
#  * - head_down: 鼻孔 y 明显大于鬐甲 y（低头）
#  * - head_up: 鼻孔 y 明显小于鬐甲 y（抬头）
#  * 说明：未使用地面检测，适合侧视稳定机位下的粗略估计
#  * @param points Dict[str, Tuple[float,float,float]]
#  * @param angles Dict[str, float]
#  * @returns str 姿态标签
#  */
def classify_posture(points: Dict[str, Tuple[float, float, float]], angles: Dict[str, float]) -> str:
    def coord(name: str) -> Optional[Tuple[float, float]]:
        v = points.get(name, (float('nan'), float('nan'), float('nan')))
        return (v[0], v[1]) if not (math.isnan(v[0]) or math.isnan(v[1])) else None

    nostril = coord('nostril')
    withers = coord('withers')

    elbow_angle = angles.get('elbow_angle', float('nan'))
    knee_angle = angles.get('knee_angle', float('nan'))
    hock_angle = angles.get('hock_angle', float('nan'))

    good_limb = []
    for a in [elbow_angle, knee_angle, hock_angle]:
        if not math.isnan(a):
            good_limb.append(120 <= a <= 170)
    limb_ok = (len(good_limb) >= 1 and sum(good_limb) / max(1, len(good_limb)) >= 0.5)

    if nostril and withers:
        dy = nostril[1] - withers[1]  # y 向下为正
        if dy > 0:  # 鼻孔低于鬐甲（低头）
            if limb_ok:
                return 'head_down'
        elif dy < 0:  # 鼻孔高于鬐甲（抬头）
            if limb_ok:
                return 'head_up'

    if limb_ok:
        return 'standing'
    return 'unknown'


# /**
#  * 在图像上绘制骨架与标签
#  * @param image np.ndarray BGR 图像
#  * @param points Dict[str, Tuple[float,float,float]] {name: (x,y,l)}
#  * @param edges List[Tuple[str,str]] 骨架边
#  * @param conf_threshold float 置信度阈值
#  * @param text Optional[str] 叠加文本（如姿态标签）
#  * @returns np.ndarray 叠加后的图像
#  */
def draw_skeleton(
    image: np.ndarray,
    points: Dict[str, Tuple[float, float, float]],
    edges: List[Tuple[str, str]] = SKELETON_EDGES,
    conf_threshold: float = 0.5,
    text: Optional[str] = None
) -> np.ndarray:
    canvas = image.copy()
    # 画边
    for a, b in edges:
        pa = points.get(a)
        pb = points.get(b)
        if not pa or not pb:
            continue
        if math.isnan(pa[0]) or math.isnan(pa[1]) or math.isnan(pb[0]) or math.isnan(pb[1]):
            continue
        if (not math.isnan(pa[2]) and pa[2] < conf_threshold) or (not math.isnan(pb[2]) and pb[2] < conf_threshold):
            continue
        cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), COLOR_EDGE, 2, cv2.LINE_AA)
    # 画点
    for name, (x, y, l) in points.items():
        if math.isnan(x) or math.isnan(y) or (not math.isnan(l) and l < conf_threshold):
            continue
        cv2.circle(canvas, (int(x), int(y)), 4, COLOR_POINT, -1, cv2.LINE_AA)
    # 文本
    if text:
        cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2, cv2.LINE_AA)
    return canvas


# /**
#  * 计算单帧有效关键点占比（x/y 非 NaN 且通过置信度过滤后）
#  * @param df pd.DataFrame DLC 结果
#  * @param frame_idx int 帧号
#  * @param bodyparts List[str]
#  * @param scorer str
#  * @returns float 取值 [0,1]
#  */
def compute_valid_point_ratio(
    df: pd.DataFrame,
    frame_idx: int,
    bodyparts: List[str],
    scorer: str
) -> float:
    total = max(1, len(bodyparts))
    valid = 0
    for bp in bodyparts:
        x_col = (scorer, bp, 'x')
        y_col = (scorer, bp, 'y')
        x = df[x_col].values[frame_idx] if x_col in df.columns else float('nan')
        y = df[y_col].values[frame_idx] if y_col in df.columns else float('nan')
        if not (math.isnan(float(x)) or math.isnan(float(y))):
            valid += 1
    return float(valid) / float(total)


# /**
#  * 安全统计函数：返回均值/标准差/缺失占比
#  * @param arr np.ndarray
#  * @returns Tuple[float, float, float]
#  */
def safe_stats(arr: np.ndarray) -> Tuple[float, float, float]:
    a = np.array(arr, dtype=float)
    mask = ~np.isnan(a)
    if mask.sum() == 0:
        return float('nan'), float('nan'), 1.0
    vals = a[mask]
    return float(np.mean(vals)), float(np.std(vals)), float(1.0 - len(vals) / len(a))


# /**
#  * 基于整段序列的简易健康评估（启发式）。
#  * 评分范围 0-100，状态分为：良好/需关注/风险。
#  * 主要依据：
#  * - 姿态分布（低头比例）
#  * - 关节角度异常占比（过小或过大）
#  * - 背线角度稳定性（标准差）
#  * - 数据质量（有效关键点占比）
#  * @param df pd.DataFrame DLC 原始（已过滤/平滑后）
#  * @param bodyparts List[str]
#  * @param scorer str
#  * @param angles_df pd.DataFrame 每帧角度与姿态（由本脚本生成）
#  * @returns Dict[str, Any] 健康报告
#  */
def evaluate_health(
    df: pd.DataFrame,
    bodyparts: List[str],
    scorer: str,
    angles_df: pd.DataFrame
) -> Dict[str, Any]:
    n = len(angles_df)
    if n == 0:
        return {
            'score': 0,
            'status': '风险',
            'reason': ['无有效帧'],
            'metrics': {}
        }

    posture = angles_df['posture'].astype(str).values
    head_down_ratio = float(np.mean(posture == 'head_down'))
    head_up_ratio = float(np.mean(posture == 'head_up'))
    standing_ratio = float(np.mean(posture == 'standing'))
    unknown_ratio = float(np.mean(~np.isin(posture, ['head_down', 'head_up', 'standing'])))

    elbow = angles_df.get('elbow_angle', pd.Series([float('nan')]*n)).values
    knee = angles_df.get('knee_angle', pd.Series([float('nan')]*n)).values
    hock = angles_df.get('hock_angle', pd.Series([float('nan')]*n)).values
    back = angles_df.get('back_angle', pd.Series([float('nan')]*n)).values

    elbow_mean, elbow_std, elbow_miss = safe_stats(elbow)
    knee_mean, knee_std, knee_miss = safe_stats(knee)
    hock_mean, hock_std, hock_miss = safe_stats(hock)
    back_mean, back_std, back_miss = safe_stats(back)

    # 角度异常（阈值可按需调整）
    def out_of_range_ratio(arr: np.ndarray, lo: float = 110, hi: float = 175) -> float:
        a = np.array(arr, dtype=float)
        mask = ~np.isnan(a)
        if mask.sum() == 0:
            return 1.0
        vals = a[mask]
        bad = np.logical_or(vals < lo, vals > hi)
        return float(np.mean(bad))

    elbow_oorr = out_of_range_ratio(elbow)
    knee_oorr = out_of_range_ratio(knee)
    hock_oorr = out_of_range_ratio(hock)
    oorr_total = np.nanmean([elbow_oorr, knee_oorr, hock_oorr])

    # 数据质量：每帧有效关键点占比 < 0.5 视为低质量帧
    low_quality = 0
    for i in range(n):
        r = compute_valid_point_ratio(df, i, bodyparts, scorer)
        if r < 0.5:
            low_quality += 1
    low_quality_ratio = float(low_quality) / float(n)

    # 评分（0-100）
    score = 100.0
    score -= head_down_ratio * 25.0
    score -= min(30.0, oorr_total * 100.0)
    score -= 10.0 if back_std > 12.0 else 0.0
    score -= low_quality_ratio * 20.0
    score = max(0.0, min(100.0, score))

    # 状态
    if score >= 80.0:
        status = '良好'
    elif score >= 60.0:
        status = '需关注'
    else:
        status = '风险'

    # 标记与建议
    flags: List[str] = []
    recs: List[str] = []
    if head_down_ratio > 0.5:
        flags.append('长期低头')
        recs.append('检查是否疲劳、采食或不适；观察是否持续超过场景需要。')
    if back_std > 12.0:
        flags.append('背线角度波动较大')
        recs.append('建议检查鞍背、背部触压敏感或步态稳定性。')
    if oorr_total > 0.2:
        flags.append('关节角度异常占比偏高')
        recs.append('关注肘/膝/飞节关节负重或活动范围；必要时安排近距离评估。')
    if low_quality_ratio > 0.2:
        flags.append('数据质量较差')
        recs.append('提升视频分辨率/光照，或调整相机视角；提高关键点置信度阈值。')

    metrics = {
        'frames': n,
        'head_down_ratio': head_down_ratio,
        'head_up_ratio': head_up_ratio,
        'standing_ratio': standing_ratio,
        'unknown_ratio': unknown_ratio,
        'elbow_mean': elbow_mean,
        'elbow_std': elbow_std,
        'knee_mean': knee_mean,
        'knee_std': knee_std,
        'hock_mean': hock_mean,
        'hock_std': hock_std,
        'back_mean': back_mean,
        'back_std': back_std,
        'elbow_out_of_range_ratio': elbow_oorr,
        'knee_out_of_range_ratio': knee_oorr,
        'hock_out_of_range_ratio': hock_oorr,
        'low_quality_ratio': low_quality_ratio
    }

    return {
        'score': round(float(score), 1),
        'status': status,
        'flags': flags,
        'recommendations': recs,
        'metrics': metrics
    }


# /**
#  * 利用 DeepLabCut Hub 模型进行单图与视频分析，并生成可视化与角度/姿态导出
#  * @param image_path str 单图路径
#  * @param video_path str 视频路径
#  * @param output_video str 输出视频路径（叠加骨架与姿态）
#  */
def run_pose_pipeline(
    image_path: str = 'horse_image.jpg',
    video_path: str = 'horse_video.mp4',
    output_video: str = 'horse_video_pose.mp4'
) -> None:
    # 仅执行关键点检测（ModelZoo → 生成 CSV），跳过后处理
    if not os.path.exists(video_path):
        return
    try:
        SpatiotemporalAdaptation(
            video_path=video_path,
            supermodel_name='superanimal_quadruped',
            videotype='mp4',
            adapt_iterations=0,
            modelfolder='',
            customized_pose_config='',
            init_weights='' 
        )
    except Exception:
        return

    # 输出生成的 CSV 路径，供后续调试
    base = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path) or '.'
    csv_candidates = glob.glob(os.path.join(video_dir, f"*{base}*DLC*.csv"))
    if not csv_candidates:
        csv_candidates = glob.glob(os.path.join(video_dir, f"*{base}*.csv"))
    if csv_candidates:
        csv_path = max(csv_candidates, key=os.path.getmtime)
        print(f"CSV: {csv_path}")
    else:
        print("未找到生成的 CSV。")


if __name__ == '__main__':
    # 默认运行：单图与单视频管线
    run_pose_pipeline(
        image_path='horse_image.jpg',
        video_path='test/VAGotrdRsWk.mp4',
        output_video='horse_video_pose.mp4'
    )