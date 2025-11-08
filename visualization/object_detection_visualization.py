"""
目标检测可视化工具 - 简化版
直接运行即可进行目标检测可视化并保存结果

用法:
python visualization/object_detection_visualization.py --input videos/111.mp4
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool.video_crop_processor import AutoCropper  # noqa: E402


def detect_and_visualize(
    input_video, output_dir="visualization/output_visualization/detection", confidence_threshold=0.6, max_frames=None
):
    """检测视频并保存可视化结果"""
    print(f"🔍 开始检测视频: {input_video}")

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
        # 初始化检测器
        detector = AutoCropper("weights/yolov11n.pt")

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
        output_video = Path(output_dir) / f"{video_name}_detection_{timestamp}.mp4"
        output_sample = Path(output_dir) / f"{video_name}_sample_{timestamp}.jpg"

        # 创建视频写入器
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_count = 0
        total_detections = 0
        process_frames = min(max_frames or total_frames, total_frames)
        sample_saved = False

        while frame_count < process_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 使用新的检测接口
            detections = detector.detect_with_details(
                frame=frame,
                target_class="person",
                confidence_threshold=confidence_threshold,
                return_format="detailed",
                sort_by="confidence",
            )

            # 可视化绘制
            annotated = visualize_detections(frame, detections, frame_info=f"Frame: {frame_count+1}/{process_frames}")

            # 保存
            writer.write(annotated)
            total_detections += len(detections)

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

        # 统计
        stats = {
            "input_video": input_video,
            "output_video": str(output_video),
            "output_sample": str(output_sample),
            "processed_frames": frame_count,
            "total_detections": total_detections,
            "avg_detections": total_detections / frame_count if frame_count > 0 else 0,
            "timestamp": timestamp,
        }

        # 保存统计
        stats_file = Path(output_dir) / f"{video_name}_stats_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print("✅ 检测完成!")
        print(f"📹 检测视频: {output_video}")
        print(f"📸 示例图片: {output_sample}")
        print(f"📊 统计文件: {stats_file}")
        print(f"📈 处理了 {frame_count} 帧，检测到 {total_detections} 个目标")

        return stats

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return None


def visualize_detections(
    frame: np.ndarray,
    detections: list,
    frame_info: str = None,
    show_confidence: bool = True,
    show_class_name: bool = True,
    show_bbox: bool = True,
    show_center: bool = True,
    bbox_thickness: int = 2,
    font_scale: float = 0.5,
    color_scheme: str = "auto",
    transparency: float = 0.7,
):
    """
    可视化检测结果的通用函数

    Args:
        frame: 输入帧
        detections: 检测结果列表
        frame_info: 帧信息文本
        show_confidence: 显示置信度
        show_class_name: 显示类别名
        show_bbox: 显示边框
        show_center: 显示中心点
        bbox_thickness: 边框厚度
        font_scale: 字体大小
        color_scheme: 颜色方案
        transparency: 透明度

    Returns:
        可视化后的帧
    """
    annotated = frame.copy()

    # 颜色生成
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
        (255, 0, 128),
    ]

    for i, detection in enumerate(detections):
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        class_name = detection["class"]

        x1, y1, x2, y2 = map(int, bbox)
        color = colors[i % len(colors)]

        # 绘制检测框
        if show_bbox:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, bbox_thickness)

        # 绘制中心点
        if show_center:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(annotated, center, 5, color, -1)

        # 构建标签
        label_parts = []
        if show_class_name:
            label_parts.append(class_name)
        if show_confidence:
            label_parts.append(f"{confidence:.2f}")

        if label_parts:
            label = " ".join(label_parts)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    # 绘制帧信息
    if frame_info:
        info_text = f"{frame_info} | Detections: {len(detections)}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated


def main():
    """主函数 - 支持丰富的检测参数配置"""
    parser = argparse.ArgumentParser(description="目标检测可视化工具 - 支持丰富参数调试")

    # === 基础参数 ===
    parser.add_argument("--input", "-i", default="videos/111.mp4", help="输入视频路径 (默认: videos/111.mp4)")
    parser.add_argument("--output", "-o", default="visualization/output_visualization/detection", help="输出目录")
    parser.add_argument("--max-frames", type=int, help="最大处理帧数 (默认: 处理全部)")

    # === 检测参数 ===
    parser.add_argument("--conf", type=float, default=0.45, help="置信度阈值 (默认: 0.6)")
    parser.add_argument("--target-class", default="person", help="目标类别 (默认: person, 使用 all 检测所有)")
    parser.add_argument("--max-detections", type=int, help="每帧最大检测数量 (默认: 无限制)")
    parser.add_argument("--min-box-area", type=float, default=100, help="最小检测框面积 (默认: 100)")
    parser.add_argument("--max-box-area", type=float, help="最大检测框面积 (默认: 无限制)")
    parser.add_argument("--crop-margin", type=float, default=0, help="裁剪边距像素 (默认: 0)")
    parser.add_argument(
        "--sort-by",
        default="confidence",
        choices=["confidence", "area", "position"],
        help="检测结果排序方式 (默认: confidence)",
    )
    parser.add_argument("--filter-overlapping", action="store_true", help="过滤重叠框 (默认: False)")
    parser.add_argument("--overlap-threshold", type=float, default=0.5, help="重叠阈值 (默认: 0.5)")

    # === 可视化参数 ===
    parser.add_argument("--show-confidence", action="store_true", default=True, help="显示置信度 (默认: True)")
    parser.add_argument("--show-class-name", action="store_true", default=True, help="显示类别名 (默认: True)")
    parser.add_argument("--show-center", action="store_true", default=True, help="显示中心点 (默认: True)")
    parser.add_argument("--bbox-thickness", type=int, default=2, help="边框厚度 (默认: 2)")
    parser.add_argument("--font-scale", type=float, default=0.5, help="字体大小 (默认: 0.5)")
    parser.add_argument(
        "--color-scheme",
        default="auto",
        choices=["auto", "class_based", "confidence_based"],
        help="颜色方案 (默认: auto)",
    )

    args = parser.parse_args()

    print("🔍 目标检测可视化工具")
    print("=" * 50)
    print("📁 配置参数:")
    print(f"   输入视频: {args.input}")
    print(f"   输出目录: {args.output}")
    print(f"   置信度阈值: {args.conf}")
    print(f"   目标类别: {args.target_class}")
    print(f"   最大检测数: {args.max_detections or '无限制'}")
    print(f"   最小面积: {args.min_box_area}")
    print(f"   最大面积: {args.max_box_area or '无限制'}")
    print(f"   排序方式: {args.sort_by}")
    print(f"   过滤重叠: {args.filter_overlapping}")
    print(f"   边框厚度: {args.bbox_thickness}")
    print(f"   字体大小: {args.font_scale}")
    print("=" * 50)

    # 执行高级检测可视化
    detect_and_visualize_advanced(
        input_video=args.input,
        output_dir=args.output,
        confidence_threshold=args.conf,
        max_frames=args.max_frames,
        target_class=args.target_class,
        max_detections=args.max_detections,
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area,
        crop_margin=args.crop_margin,
        sort_by=args.sort_by,
        filter_overlapping=args.filter_overlapping,
        overlap_threshold=args.overlap_threshold,
        show_confidence=args.show_confidence,
        show_class_name=args.show_class_name,
        show_center=args.show_center,
        bbox_thickness=args.bbox_thickness,
        font_scale=args.font_scale,
        color_scheme=args.color_scheme,
    )


def detect_and_visualize_advanced(
    input_video: str,
    output_dir: str = "visualization/output_visualization/detection",
    confidence_threshold: float = 0.6,
    max_frames: int = None,
    target_class: str = "person",
    max_detections: int = None,
    min_box_area: float = 100,
    max_box_area: float = None,
    crop_margin: float = 0,
    sort_by: str = "confidence",
    filter_overlapping: bool = False,
    overlap_threshold: float = 0.5,
    show_confidence: bool = True,
    show_class_name: bool = True,
    show_center: bool = True,
    bbox_thickness: int = 2,
    font_scale: float = 0.5,
    color_scheme: str = "auto",
):
    """
    高级检测可视化函数，支持丰富的参数配置
    """
    print(f"🔍 开始高级检测处理: {input_video}")

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
        # 初始化检测器
        detector = AutoCropper("weights/yolov11n.pt")

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
        output_video = Path(output_dir) / f"{video_name}_detection_{timestamp}.mp4"
        output_sample = Path(output_dir) / f"{video_name}_sample_{timestamp}.jpg"

        # 创建视频写入器
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_count = 0
        total_detections = 0
        process_frames = min(max_frames or total_frames, total_frames)
        sample_saved = False

        while frame_count < process_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 使用高级检测接口
            detections = detector.detect_with_details(
                frame=frame,
                target_class=target_class if target_class != "all" else None,
                confidence_threshold=confidence_threshold,
                max_detections=max_detections,
                min_box_area=min_box_area,
                max_box_area=max_box_area,
                crop_margin=crop_margin,
                return_format="detailed",
                filter_overlapping=filter_overlapping,
                overlap_threshold=overlap_threshold,
                sort_by=sort_by,
            )

            # 高级可视化
            annotated = visualize_detections(
                frame=frame,
                detections=detections,
                frame_info=f"Frame: {frame_count+1}/{process_frames}",
                show_confidence=show_confidence,
                show_class_name=show_class_name,
                show_bbox=True,
                show_center=show_center,
                bbox_thickness=bbox_thickness,
                font_scale=font_scale,
                color_scheme=color_scheme,
            )

            # 保存
            writer.write(annotated)
            total_detections += len(detections)

            # 保存示例帧
            if not sample_saved and frame_count == process_frames // 2:
                cv2.imwrite(str(output_sample), annotated)
                sample_saved = True

            frame_count += 1

            # 进度
            if frame_count % 50 == 0:
                progress = frame_count / process_frames * 100
                print(f"⏳ 进度: {progress:.1f}% ({frame_count}/{process_frames}) - 当前帧检测: {len(detections)}")

        cap.release()
        writer.release()

        # 统计
        stats = {
            "input_video": input_video,
            "output_video": str(output_video),
            "output_sample": str(output_sample),
            "processed_frames": frame_count,
            "total_detections": total_detections,
            "avg_detections": total_detections / frame_count if frame_count > 0 else 0,
            "settings": {
                "confidence_threshold": confidence_threshold,
                "target_class": target_class,
                "max_detections": max_detections,
                "min_box_area": min_box_area,
                "max_box_area": max_box_area,
                "sort_by": sort_by,
                "filter_overlapping": filter_overlapping,
            },
            "timestamp": timestamp,
        }

        # 保存统计
        stats_file = Path(output_dir) / f"{video_name}_stats_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print("✅ 检测完成!")
        print(f"📹 检测视频: {output_video}")
        print(f"📸 示例图片: {output_sample}")
        print(f"📊 统计文件: {stats_file}")
        print(f"📈 处理了 {frame_count} 帧，检测到 {total_detections} 个目标")

        return stats

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
