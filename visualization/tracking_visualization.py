"""
目标跟踪可视化工具 - 简化版
直接运行即可进行目标跟踪可视化并保存结果

用法:
python visualization/tracking_visualization.py --input videos/111.mp4
"""

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool.tracker import TrackingConfig, VideoTracker  # noqa: E402
from tool.video_crop_processor import AutoCropper  # noqa: E402


def track_and_visualize(
    input_video,
    output_dir="visualization/output_visualization/tracking",
    confidence_threshold=0.3,
    max_frames=None,
    trail_length=30,
):
    """跟踪视频并保存可视化结果"""
    print(f"🎯 开始跟踪视频: {input_video}")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 检查文件
    if not Path(input_video).exists():
        print(f"❌ 视频文件不存在: {input_video}")
        return None

    if not Path("weights/yolov11n.pt").exists():
        print("❌ 模型文件不存在: weights/yolov11n.pt")
        return None

    try:
        # 创建检测器和跟踪器
        detector = AutoCropper("weights/yolov11n.pt")
        config = TrackingConfig(
            enable_tracking=True,
            tracker_type="bytetrack",
            track_buffer=30,
            track_low_thresh=confidence_threshold,  # 🔧 修复：使用传入的置信度阈值
            track_high_thresh=confidence_threshold * 1.2,  # 高置信度阈值稍高一些
            new_track_thresh=confidence_threshold,  # 新轨迹阈值
        )
        tracker = VideoTracker(config)

        # 打开视频
        cap = cv2.VideoCapture(input_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 视频: {width}x{height}, {fps}fps, {total_frames}帧")

        # 输出文件
        video_name = Path(input_video).stem
        timestamp = datetime.now().strftime("%H%M%S")
        output_video = Path(output_dir) / f"{video_name}_tracking_{timestamp}.mp4"
        output_sample = Path(output_dir) / f"{video_name}_sample_{timestamp}.jpg"
        output_trails = Path(output_dir) / f"{video_name}_trails_{timestamp}.jpg"

        # 创建视频写入器
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        # 轨迹存储
        trails = defaultdict(lambda: deque(maxlen=trail_length))
        track_ids = set()
        trails_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        def get_color(track_id):
            """为ID分配颜色"""
            hue = int((track_id * 137) % 180)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            return (int(color[0]), int(color[1]), int(color[2]))

        frame_count = 0
        total_detections = 0
        total_tracks = 0
        process_frames = min(max_frames or total_frames, total_frames)
        sample_saved = False

        while frame_count < process_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 先检测
            detections = detector.detect_with_details(
                frame=frame, target_class="person", confidence_threshold=confidence_threshold, return_format="detailed"
            )

            # 2. 转换为跟踪格式
            tracking_detections = []
            for det in detections:
                bbox = det["bbox"]
                conf = det["confidence"]
                tracking_detections.append((*bbox, conf, 0))  # (x1,y1,x2,y2,conf,class_id)

            # 3. 执行跟踪
            tracks = tracker.update(
                detections=tracking_detections,
                frame_id=frame_count,
                frame_size=(width, height),
                track_confidence_threshold=confidence_threshold,
                enable_track_smoothing=True,
            )

            # 4. 可视化
            annotated = visualize_tracking(
                frame=frame,
                tracks=tracks,
                trails=trails,
                trails_canvas=trails_canvas,
                get_color_func=get_color,
                frame_info=f"Frame: {frame_count+1}",
            )

            # 统计
            current_detections = len(detections)
            current_tracks = len(tracks)
            total_detections += current_detections
            total_tracks += current_tracks

            if tracks:
                track_ids.update(tracks.keys())

            # 保存
            writer.write(annotated)

            # 保存示例帧
            if not sample_saved and frame_count == process_frames // 2:
                cv2.imwrite(str(output_sample), annotated)
                sample_saved = True

            frame_count += 1

            # 进度
            if frame_count % 50 == 0:
                progress = frame_count / process_frames * 100
                print(f"⏳ 进度: {progress:.1f}% ({frame_count}/{process_frames})")

        cap.release()
        writer.release()

        # 保存轨迹图
        cv2.imwrite(str(output_trails), trails_canvas)

        # 统计
        stats = {
            "input_video": input_video,
            "output_video": str(output_video),
            "output_sample": str(output_sample),
            "output_trails": str(output_trails),
            "processed_frames": frame_count,
            "total_detections": total_detections,
            "total_tracks": total_tracks,
            "unique_ids": len(track_ids),
            "avg_detections": total_detections / frame_count if frame_count > 0 else 0,
            "avg_tracks": total_tracks / frame_count if frame_count > 0 else 0,
            "timestamp": timestamp,
        }

        # 保存统计
        stats_file = Path(output_dir) / f"{video_name}_stats_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print("✅ 跟踪完成!")
        print(f"📹 跟踪视频: {output_video}")
        print(f"📸 示例图片: {output_sample}")
        print(f"🛤️  轨迹图片: {output_trails}")
        print(f"📊 统计文件: {stats_file}")
        print(
            f"📈 处理了 {frame_count} 帧，检测 {total_detections} 次，跟踪 {total_tracks} 次，{len(track_ids)} 个唯一ID"
        )

        return stats

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return None


def visualize_tracking(
    frame: np.ndarray,
    tracks: dict,
    trails: dict,
    trails_canvas: np.ndarray,
    get_color_func,
    frame_info: str = None,
    show_trail: bool = True,
    show_velocity: bool = False,
    show_track_id: bool = True,
    trail_thickness: int = 2,
    fade_trail: bool = True,
    color_per_track: bool = True,
):
    """
    可视化跟踪结果的通用函数

    Args:
        frame: 输入帧
        tracks: 跟踪结果字典
        trails: 轨迹历史
        trails_canvas: 轨迹画布
        get_color_func: 获取颜色的函数
        frame_info: 帧信息
        show_trail: 显示轨迹
        show_velocity: 显示速度
        show_track_id: 显示跟踪ID
        trail_thickness: 轨迹厚度
        fade_trail: 轨迹淡化
        color_per_track: 每个轨迹不同颜色

    Returns:
        可视化后的帧
    """
    annotated = frame.copy()

    for track_id, track_info in tracks.items():
        if isinstance(track_info, dict):
            bbox = track_info.get("bbox")
            confidence = track_info.get("confidence", 1.0)
            track_info.get("state", "active")
        else:
            # 兼容旧格式 (x1,y1,x2,y2)
            bbox = track_info
            confidence = 1.0

        if not bbox:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 获取颜色
        color = get_color_func(track_id)

        # 更新轨迹
        trails[track_id].append(center)

        # 绘制轨迹
        if show_trail and len(trails[track_id]) > 1:
            for i in range(1, len(trails[track_id])):
                pt1 = trails[track_id][i - 1]
                pt2 = trails[track_id][i]

                if fade_trail:
                    alpha = i / len(trails[track_id])
                    thickness = max(1, int(trail_thickness * alpha))
                else:
                    thickness = trail_thickness

                cv2.line(annotated, pt1, pt2, color, thickness)
                cv2.line(trails_canvas, pt1, pt2, color, 2)

        # 绘制检测框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, center, 5, color, -1)

        # 标签
        label_parts = []
        if show_track_id:
            label_parts.append(f"ID:{track_id}")
        label_parts.append(f"{confidence:.2f}")
        if isinstance(track_info, dict) and "class" in track_info:
            label_parts.append(track_info["class"])

        label = " ".join(label_parts)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 绘制帧信息
    if frame_info:
        active_tracks = [t for t in tracks.values() if isinstance(t, dict) and t.get("state") == "active"]
        info_text = f"{frame_info} | Det: {len(tracks)} | " f"Tracks: {len(active_tracks)} | IDs: {len(tracks)}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated


def main():
    """主函数 - 支持丰富的跟踪参数配置"""
    parser = argparse.ArgumentParser(description="目标跟踪可视化工具 - 支持丰富参数调试")

    # === 基础参数 ===
    parser.add_argument("--input", "-i", default="videos/111.mp4", help="输入视频路径 (默认: videos/111.mp4)")
    parser.add_argument("--output", "-o", default="visualization/output_visualization/tracking", help="输出目录")
    parser.add_argument("--max-frames", type=int, help="最大处理帧数 (默认: 处理全部)")

    # === 检测参数 ===
    parser.add_argument("--conf", type=float, default=0.45, help="检测置信度阈值 (默认: 0.3)")
    parser.add_argument("--target-class", default="person", help="目标类别 (默认: person, 使用 all 检测所有)")
    parser.add_argument("--max-detections", type=int, help="每帧最大检测数量 (默认: 无限制)")
    parser.add_argument("--min-box-area", type=float, default=100, help="最小检测框面积 (默认: 100)")
    parser.add_argument("--max-box-area", type=float, help="最大检测框面积 (默认: 无限制)")

    # === 跟踪参数 ===
    parser.add_argument("--track-conf", type=float, default=0.25, help="跟踪置信度阈值 (默认: 0.25)")
    parser.add_argument(
        "--association-metric",
        default="cosine",
        choices=["cosine", "euclidean", "iou"],
        help="关联度量方式 (默认: cosine)",
    )
    parser.add_argument("--enable-kalman", action="store_true", default=True, help="启用卡尔曼滤波 (默认: True)")
    parser.add_argument("--max-age", type=int, default=30, help="最大追踪年龄 (默认: 30)")
    parser.add_argument("--min-hits", type=int, default=3, help="最小命中次数 (默认: 3)")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU阈值 (默认: 0.3)")

    # === 可视化参数 ===
    parser.add_argument("--trail-length", type=int, default=30, help="轨迹长度 (默认: 30)")
    parser.add_argument("--show-track-id", action="store_true", default=True, help="显示跟踪ID (默认: True)")
    parser.add_argument("--show-confidence", action="store_true", default=True, help="显示置信度 (默认: True)")
    parser.add_argument("--show-trails", action="store_true", default=True, help="显示运动轨迹 (默认: True)")
    parser.add_argument("--fade-trail", action="store_true", default=True, help="渐变轨迹 (默认: True)")
    parser.add_argument("--trail-thickness", type=int, default=2, help="轨迹厚度 (默认: 2)")
    parser.add_argument("--bbox-thickness", type=int, default=2, help="边框厚度 (默认: 2)")
    parser.add_argument("--font-scale", type=float, default=0.5, help="字体大小 (默认: 0.5)")
    parser.add_argument(
        "--color-scheme",
        default="id_based",
        choices=["id_based", "confidence_based", "class_based"],
        help="颜色方案 (默认: id_based)",
    )

    # === 输出参数 ===
    parser.add_argument("--save-trails", action="store_true", default=True, help="保存轨迹图 (默认: True)")
    parser.add_argument("--save-stats", action="store_true", default=True, help="保存统计信息 (默认: True)")

    args = parser.parse_args()

    print("🎯 目标跟踪可视化工具")
    print("=" * 50)
    print("📁 配置参数:")
    print(f"   输入视频: {args.input}")
    print(f"   输出目录: {args.output}")
    print(f"   检测置信度: {args.conf}")
    print(f"   跟踪置信度: {args.track_conf}")
    print(f"   目标类别: {args.target_class}")
    print(f"   关联度量: {args.association_metric}")
    print(f"   卡尔曼滤波: {args.enable_kalman}")
    print(f"   轨迹长度: {args.trail_length}")
    print(f"   最大年龄: {args.max_age}")
    print(f"   最小命中: {args.min_hits}")
    print(f"   IoU阈值: {args.iou_threshold}")
    print(f"   颜色方案: {args.color_scheme}")
    print("=" * 50)

    # 执行高级跟踪可视化
    track_and_visualize_advanced(
        input_video=args.input,
        output_dir=args.output,
        confidence_threshold=args.conf,
        max_frames=args.max_frames,
        target_class=args.target_class,
        max_detections=args.max_detections,
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area,
        track_confidence_threshold=args.track_conf,
        association_metric=args.association_metric,
        enable_kalman_filter=args.enable_kalman,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        trail_length=args.trail_length,
        show_track_id=args.show_track_id,
        show_confidence=args.show_confidence,
        show_trails=args.show_trails,
        fade_trail=args.fade_trail,
        trail_thickness=args.trail_thickness,
        bbox_thickness=args.bbox_thickness,
        font_scale=args.font_scale,
        color_scheme=args.color_scheme,
        save_trails=args.save_trails,
        save_stats=args.save_stats,
    )


def track_and_visualize_advanced(
    input_video: str,
    output_dir: str = "visualization/output_visualization/tracking",
    confidence_threshold: float = 0.3,
    max_frames: int = None,
    target_class: str = "person",
    max_detections: int = None,
    min_box_area: float = 100,
    max_box_area: float = None,
    track_confidence_threshold: float = 0.25,
    association_metric: str = "cosine",
    enable_kalman_filter: bool = True,
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    trail_length: int = 30,
    show_track_id: bool = True,
    show_confidence: bool = True,
    show_trails: bool = True,
    fade_trail: bool = True,
    trail_thickness: int = 2,
    bbox_thickness: int = 2,
    font_scale: float = 0.5,
    color_scheme: str = "id_based",
    save_trails: bool = True,
    save_stats: bool = True,
):
    """
    高级跟踪可视化函数，支持丰富的参数配置
    """
    print(f"🎯 开始高级跟踪处理: {input_video}")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 检查文件
    if not Path(input_video).exists():
        print(f"❌ 视频文件不存在: {input_video}")
        return None

    if not Path("weights/yolov11n.pt").exists():
        print("❌ 模型文件不存在: weights/yolov11n.pt")
        return None

    try:
        # 初始化检测器和跟踪器
        detector = AutoCropper("weights/yolov11n.pt")

        # 创建跟踪配置（映射参数到正确的配置名称）
        tracking_config = TrackingConfig(
            enable_tracking=True,
            tracker_type="bytetrack",
            track_buffer=max_age,  # 使用 track_buffer 替代 max_age
            match_threshold=iou_threshold,  # 使用 match_threshold 替代 iou_threshold
            min_box_area=min_box_area,
            track_high_thresh=track_confidence_threshold,  # 高置信度阈值
            track_low_thresh=track_confidence_threshold * 0.5,  # 低置信度阈值
            new_track_thresh=track_confidence_threshold,  # 新轨迹阈值
            track_lost_thresh=max_age,  # 轨迹丢失阈值
        )
        tracker = VideoTracker(tracking_config)

        # 打开视频
        cap = cv2.VideoCapture(input_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 视频: {width}x{height}, {fps}fps, {total_frames}帧")

        # 输出文件
        video_name = Path(input_video).stem
        timestamp = datetime.now().strftime("%H%M%S")
        output_video = Path(output_dir) / f"{video_name}_tracking_{timestamp}.mp4"
        output_sample = Path(output_dir) / f"{video_name}_sample_{timestamp}.jpg"
        output_trails = Path(output_dir) / f"{video_name}_trails_{timestamp}.jpg" if save_trails else None

        # 创建视频写入器
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        # 轨迹画布
        trails_canvas = np.zeros((height, width, 3), dtype=np.uint8) if save_trails else None

        frame_count = 0
        total_detections = 0
        total_tracks = 0
        process_frames = min(max_frames or total_frames, total_frames)
        sample_saved = False
        track_history = {}

        while frame_count < process_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 高级检测
            detections = detector.detect_with_details(
                frame=frame,
                target_class=target_class if target_class != "all" else None,
                confidence_threshold=confidence_threshold,
                max_detections=max_detections,
                min_box_area=min_box_area,
                max_box_area=max_box_area,
                return_format="detailed",
                filter_overlapping=True,
                overlap_threshold=0.5,
                sort_by="confidence",
            )

            # 高级跟踪更新
            tracks = tracker.update(
                detections=detections,
                frame_id=frame_count,
                track_confidence_threshold=track_confidence_threshold,
                association_metric=association_metric,
                enable_kalman_filter=enable_kalman_filter,
            )

            # 更新轨迹历史
            for track_id, track_data in tracks.items():
                if track_id not in track_history:
                    track_history[track_id] = []

                if isinstance(track_data, dict) and "bbox" in track_data:
                    bbox = track_data["bbox"]
                    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    track_history[track_id].append(center)

                    # 限制轨迹长度
                    if len(track_history[track_id]) > trail_length:
                        track_history[track_id] = track_history[track_id][-trail_length:]

            # 高级可视化（调整参数名称以匹配函数签名）
            def get_color_for_track(track_id):
                """根据颜色方案获取轨迹颜色"""
                if color_scheme == "id_based":
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    return colors[track_id % len(colors)]
                elif color_scheme == "confidence_based":
                    # 基于置信度的颜色（需要从tracks获取置信度）
                    return (0, 255, 0)  # 默认绿色
                else:
                    return (255, 255, 255)  # 默认白色

            annotated = visualize_tracking(
                frame=frame,
                tracks=tracks,
                trails=track_history if show_trails else {},
                trails_canvas=trails_canvas,
                get_color_func=get_color_for_track,
                frame_info=f"Frame: {frame_count+1}/{process_frames}",
                show_trail=show_trails,
                show_velocity=False,  # 这里使用false，因为我们还没实现速度计算
                show_track_id=show_track_id,
                trail_thickness=trail_thickness,
                fade_trail=fade_trail,
                color_per_track=(color_scheme == "id_based"),
            )

            # 保存
            writer.write(annotated)
            total_detections += len(detections)
            total_tracks += len(tracks)

            # 保存示例帧
            if not sample_saved and frame_count == process_frames // 2:
                cv2.imwrite(str(output_sample), annotated)
                sample_saved = True

            frame_count += 1

            # 进度
            if frame_count % 50 == 0:
                progress = frame_count / process_frames * 100
                active_tracks = len([t for t in tracks.values() if isinstance(t, dict) and t.get("state") == "active"])
                print(
                    f"⏳ 进度: {progress:.1f}% ({frame_count}/{process_frames}) - "
                    f"检测: {len(detections)}, 活跃轨迹: {active_tracks}"
                )

        cap.release()
        writer.release()

        # 保存轨迹图
        if save_trails and trails_canvas is not None:
            cv2.imwrite(str(output_trails), trails_canvas)

        # 统计
        stats = {
            "input_video": input_video,
            "output_video": str(output_video),
            "output_sample": str(output_sample),
            "output_trails": str(output_trails) if output_trails else None,
            "processed_frames": frame_count,
            "total_detections": total_detections,
            "total_tracks": total_tracks,
            "unique_track_ids": len(track_history),
            "avg_detections": total_detections / frame_count if frame_count > 0 else 0,
            "avg_tracks": total_tracks / frame_count if frame_count > 0 else 0,
            "detection_settings": {
                "confidence_threshold": confidence_threshold,
                "target_class": target_class,
                "max_detections": max_detections,
                "min_box_area": min_box_area,
                "max_box_area": max_box_area,
            },
            "tracking_settings": {
                "track_confidence_threshold": track_confidence_threshold,
                "association_metric": association_metric,
                "enable_kalman_filter": enable_kalman_filter,
                "max_age": max_age,
                "min_hits": min_hits,
                "iou_threshold": iou_threshold,
            },
            "visualization_settings": {
                "trail_length": trail_length,
                "color_scheme": color_scheme,
                "fade_trail": fade_trail,
                "trail_thickness": trail_thickness,
            },
            "timestamp": timestamp,
        }

        # 保存统计
        if save_stats:
            stats_file = Path(output_dir) / f"{video_name}_stats_{timestamp}.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"📊 统计文件: {stats_file}")

        print("✅ 跟踪完成!")
        print(f"📹 跟踪视频: {output_video}")
        print(f"📸 示例图片: {output_sample}")
        if output_trails:
            print(f"🛤️ 轨迹图片: {output_trails}")
        print(f"📈 处理了 {frame_count} 帧")
        print(f"🔍 总检测数: {total_detections} (平均 {total_detections/frame_count:.1f}/帧)")
        print(f"🎯 总跟踪数: {total_tracks} (平均 {total_tracks/frame_count:.1f}/帧)")
        print(f"🏷️ 唯一ID数: {len(track_history)}")

        return stats

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
