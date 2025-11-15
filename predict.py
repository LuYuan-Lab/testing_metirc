import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model.ResNetModel import R2Dmodel
from tool.dataset import VideoDataset
from tool.video_crop_processor import AutoCropper


class SequenceVideoPredictor:
    """专注于视频序列检测的预测器"""

    def __init__(
        self,
        model_path: str,
        reference_embeddings_path: str,
        yolo_model_path: str = "weights/yolov11n.pt",
        embedding_dim: int = 128,
        device: str = "auto",
    ):
        """
        初始化序列预测器

        Args:
            model_path: 训练好的模型权重路径
            reference_embeddings_path: 参考特征向量路径
            yolo_model_path: YOLO模型路径
            embedding_dim: 特征向量维度
            device: 计算设备
        """
        # 设置设备
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        )
        print(f"Using device: {self.device}")

        # 加载行为识别模型
        self.model = R2Dmodel(embedding_dim=embedding_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 加载YOLO检测器
        self.detector = AutoCropper(model_path=yolo_model_path, conf_thres=0.25, target_class="person")
        print("✅ YOLO detector loaded successfully")

        # 加载参考特征向量
        self._load_reference_embeddings(reference_embeddings_path)

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_reference_embeddings(self, path: str):
        """加载参考特征向量"""
        print(f"Loading reference embeddings from {path}")
        data = torch.load(path)
        self.reference_embeddings = data["embeddings"].to(self.device)
        self.reference_labels = data["labels"].to(self.device)

        # 获取类别名称
        temp_dataset = VideoDataset(data_root="data", mode="train", num_frames=30)
        class_names = list(temp_dataset.class_to_idx.keys())

        # 中文到英文的映射
        display_names = {
            "正常": "Normal",
            "举手": "Hand Raise",
            "手机": "Phone",
            "站立": "Standing",
            "左右看": "Looking Around",
        }

        self.class_names = [display_names.get(name, name) for name in class_names]
        print(f"Loaded {len(self.reference_embeddings)} reference embeddings")
        print(f"Classes: {self.class_names}")

    def _filter_detection_boxes(
        self, detection_results, iou_threshold: float = 0.5
    ) -> List[Tuple[np.ndarray, float, int]]:
        """
        过滤检测框，去除重复框但保留多个目标

        Args:
            detection_results: YOLO检测结果
            iou_threshold: IoU阈值，用于判断是否为同一目标

        Returns:
            过滤后的检测框列表 [(bbox, confidence, track_id), ...]
        """
        all_boxes = []
        all_confidences = []

        # 收集所有person类别的检测框
        for result in detection_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0:  # person类别
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        all_boxes.append([x1, y1, x2, y2])
                        all_confidences.append(conf)

        if len(all_boxes) == 0:
            return []

        all_boxes = np.array(all_boxes)
        all_confidences = np.array(all_confidences)

        # 应用自定义NMS，保留多个目标
        filtered_indices = self._multi_target_nms(all_boxes, all_confidences, iou_threshold)

        # 构建结果
        filtered_boxes = []
        for i, idx in enumerate(filtered_indices):
            bbox = all_boxes[idx].astype(int)
            confidence = all_confidences[idx]
            track_id = i  # 简单的跟踪ID分配
            filtered_boxes.append((bbox, confidence, track_id))

        return filtered_boxes

    def _multi_target_nms(self, boxes: np.ndarray, confidences: np.ndarray, iou_threshold: float) -> List[int]:
        """
        多目标非最大抑制算法
        与传统NMS不同，这个算法会保留不同位置的多个目标
        """
        if len(boxes) == 0:
            return []

        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # 按置信度排序
        order = confidences.argsort()[::-1]

        keep = []
        while len(order) > 0:
            # 取置信度最高的框
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # 计算与其他框的IoU
            ious = self._compute_iou(boxes[i], boxes[order[1:]], areas[i], areas[order[1:]])

            # 保留IoU小于阈值的框（不同目标）
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _compute_iou(self, box1: np.ndarray, boxes: np.ndarray, area1: float, areas: np.ndarray) -> np.ndarray:
        """计算单个框与多个框的IoU"""
        # 计算交集
        xx1 = np.maximum(box1[0], boxes[:, 0])
        yy1 = np.maximum(box1[1], boxes[:, 1])
        xx2 = np.minimum(box1[2], boxes[:, 2])
        yy2 = np.minimum(box1[3], boxes[:, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        # 计算并集
        union = area1 + areas - intersection

        # 计算IoU
        ious = intersection / (union + 1e-8)
        return ious

    def predict_sequence(
        self,
        video_path: str,
        segment_duration: float = 3.0,
        overlap_ratio: float = 0.3,
        frames_per_segment: int = 30,
        confidence_threshold: float = 0.4,
        max_segments: Optional[int] = None,
    ) -> Dict:
        """
        对视频进行序列预测

        Args:
            video_path: 视频文件路径
            segment_duration: 每个时间段的持续时间（秒）
            overlap_ratio: 相邻时间段的重叠比例（0-1）
            frames_per_segment: 每个时间段采样的帧数
            confidence_threshold: 置信度阈值
            max_segments: 最大段数限制（用于测试）

        Returns:
            包含序列预测结果的字典
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / fps

        print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {video_duration:.1f}s duration")
        print(f"Segment settings: {segment_duration}s per segment, {frames_per_segment} frames per segment")

        # 计算时间段参数
        frames_per_segment_actual = int(segment_duration * fps)
        overlap_frames = int(frames_per_segment_actual * overlap_ratio)
        stride_frames = frames_per_segment_actual - overlap_frames

        # 检测统一的裁剪区域
        crop_rect = self._detect_crop_region(cap, total_frames)

        # 生成时间段
        segments = self._generate_segments(total_frames, frames_per_segment_actual, stride_frames, fps)

        if max_segments and len(segments) > max_segments:
            segments = segments[:max_segments]
            print(f"Limited to first {max_segments} segments for testing")

        # 预测每个段落
        predictions = self._predict_segments(cap, segments, frames_per_segment, crop_rect, confidence_threshold)

        cap.release()

        # 生成结果
        result = {
            "video_path": video_path,
            "video_info": {"duration": video_duration, "fps": fps, "total_frames": total_frames},
            "segment_settings": {
                "segment_duration": segment_duration,
                "frames_per_segment": frames_per_segment,
                "overlap_ratio": overlap_ratio,
                "confidence_threshold": confidence_threshold,
            },
            "summary": self._generate_summary(predictions, len(segments)),
            "predictions": predictions,
        }

        return result

    def _detect_crop_region(self, cap, total_frames):
        """检测统一的裁剪区域"""
        print("Detecting unified crop region...")
        sample_frames = min(20, total_frames)
        sample_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

        all_crops = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 使用YOLO检测并过滤检测框
                detection_results = self.detector.model(rgb_frame, conf=0.25, verbose=False)
                filtered_boxes = self._filter_detection_boxes(detection_results, iou_threshold=0.5)

                # 如果有多个目标，选择面积最大的（通常是主要目标）
                if filtered_boxes:
                    if len(filtered_boxes) == 1:
                        bbox, _, _ = filtered_boxes[0]
                        detected_crop = tuple(bbox)
                        all_crops.append(detected_crop)
                    else:
                        # 多个目标时，选择面积最大的作为主要裁剪区域
                        best_bbox = None
                        max_area = 0
                        for bbox in filtered_boxes:
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if area > max_area:
                                max_area = area
                                best_bbox = bbox
                        if best_bbox is not None:
                            detected_crop = tuple(best_bbox)
                            all_crops.append(detected_crop)
                            print(
                                f"  Frame {idx}: Found {len(filtered_boxes)} targets, "
                                f"selected largest (area={max_area:.0f})"
                            )

        if all_crops:
            crop_rect = tuple(np.mean(all_crops, axis=0).astype(int))
            print(f"Unified crop region: {crop_rect}")
            return crop_rect
        else:
            print("No person detected, using full frame")
            return None

    def _generate_segments(self, total_frames, frames_per_segment_actual, stride_frames, fps):
        """生成时间段"""
        segments = []
        start_frame = 0
        segment_id = 0

        while start_frame + frames_per_segment_actual <= total_frames:
            end_frame = start_frame + frames_per_segment_actual
            start_time = start_frame / fps
            end_time = end_frame / fps

            segments.append(
                {
                    "id": segment_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

            start_frame += stride_frames
            segment_id += 1

        print(f"Generated {len(segments)} time segments")
        return segments

    def _predict_segments(self, cap, segments, frames_per_segment, crop_rect, confidence_threshold):
        """预测所有段落"""
        predictions = []

        for segment in tqdm(segments, desc="Processing segments"):
            try:
                # 提取段落帧
                frames = self._extract_frames(cap, segment, frames_per_segment, crop_rect)

                if len(frames) < frames_per_segment:
                    continue

                # 预测
                prediction = self._predict_single_segment(frames, segment, confidence_threshold)

                if prediction:
                    predictions.append(prediction)
                    print(
                        f"Segment {segment['id']:2d}: "
                        f"{segment['start_time']:5.1f}-{segment['end_time']:5.1f}s -> "
                        f"{prediction['predicted_class']:12s} ({prediction['confidence']:.3f})"
                    )
                else:
                    print(
                        f"Segment {segment['id']:2d}: "
                        f"{segment['start_time']:5.1f}-{segment['end_time']:5.1f}s -> "
                        f"Low confidence"
                    )

            except Exception as e:
                print(f"Error processing segment {segment['id']}: {e}")
                continue

        return predictions

    def _extract_frames(self, cap, segment, frames_per_segment, crop_rect):
        """从时间段中提取帧"""
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]

        frame_indices = np.linspace(start_frame, end_frame - 1, frames_per_segment, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # 应用裁剪
            if crop_rect is not None:
                x1, y1, x2, y2 = crop_rect
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                frame = frame[y1:y2, x1:x2]

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames

    def _predict_single_segment(self, frames, segment, confidence_threshold):
        """预测单个段落"""
        # 预处理帧
        processed_frames = []
        for frame in frames:
            pil_frame = Image.fromarray(frame)
            transformed = self.transform(pil_frame)
            processed_frames.append(transformed)

        # 构造张量
        video_tensor = torch.stack(processed_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        video_tensor = video_tensor.to(self.device)

        # 特征提取
        with torch.no_grad():
            features = self.model(video_tensor).squeeze(0)

        # 相似度计算
        target_norm = torch.nn.functional.normalize(features.unsqueeze(0), p=2, dim=-1)
        reference_norm = torch.nn.functional.normalize(self.reference_embeddings, p=2, dim=-1)
        similarities = torch.matmul(target_norm, reference_norm.t()).squeeze(0).cpu().numpy()

        # 各类别平均相似度
        class_similarities = {}
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = self.reference_labels.cpu().numpy() == class_idx
            if np.any(class_mask):
                class_similarities[class_name] = float(np.mean(similarities[class_mask]))

        # 预测类别
        predicted_class = max(class_similarities, key=class_similarities.get)
        confidence = class_similarities[predicted_class]

        # 检查置信度
        if confidence >= confidence_threshold:
            return {
                "segment_id": segment["id"],
                "time_range": f"{segment['start_time']:.1f}s - {segment['end_time']:.1f}s",
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_similarities": class_similarities,
            }

        return None

    def _generate_summary(self, predictions, total_segments):
        """生成汇总统计"""
        if not predictions:
            return {
                "dominant_behavior": "Unknown",
                "total_confident_predictions": 0,
                "total_segments": total_segments,
                "behavior_distribution": {},
                "average_confidence": 0.0,
                "coverage_ratio": 0.0,
            }

        # 统计行为分布
        class_counts = {}
        total_confidence = 0

        for pred in predictions:
            cls = pred["predicted_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
            total_confidence += pred["confidence"]

        dominant_class = max(class_counts, key=class_counts.get)
        avg_confidence = total_confidence / len(predictions)
        coverage_ratio = len(predictions) / total_segments

        return {
            "dominant_behavior": dominant_class,
            "total_confident_predictions": len(predictions),
            "total_segments": total_segments,
            "behavior_distribution": class_counts,
            "average_confidence": avg_confidence,
            "coverage_ratio": coverage_ratio,
        }

    def create_visualization_video(self, video_path: str, sequence_results: Dict, output_path: str) -> str:
        """
        创建序列预测的可视化视频

        Args:
            video_path: 输入视频路径
            sequence_results: 序列预测结果
            output_path: 输出视频路径

        Returns:
            输出视频路径
        """
        print("🎬 Creating sequence visualization video...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 解析预测结果
        predictions = sequence_results.get("predictions", [])
        video_info = sequence_results.get("video_info", {})

        # 创建时间到预测的映射
        time_predictions = {}
        for pred in predictions:
            start_time = int(pred["start_time"])
            end_time = int(pred["end_time"])
            for t in range(start_time, end_time + 1):
                time_predictions[t] = {
                    "class": pred["predicted_class"],
                    "confidence": pred["confidence"],
                    "segment_info": pred["time_range"],
                }

        # 颜色映射
        color_map = {
            "Hand Raise": (0, 255, 255),  # 黄色
            "Phone": (0, 0, 255),  # 红色
            "Standing": (255, 0, 255),  # 紫色
            "Looking Around": (0, 165, 255),  # 橙色
            "Normal": (0, 255, 0),  # 绿色
        }

        try:
            frame_count = 0
            with tqdm(total=total_frames, desc="Processing video frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_time = int(frame_count / fps)
                    current_prediction = time_predictions.get(current_time)

                    # 人体检测并过滤检测框
                    detection_results = self.detector.model(frame, conf=0.25, verbose=False)
                    filtered_boxes = self._filter_detection_boxes(detection_results, iou_threshold=0.5)

                    # 绘制检测框和预测结果
                    annotated_frame = frame.copy()

                    # 绘制过滤后的检测框
                    for bbox, conf, track_id in filtered_boxes:
                        x1, y1, x2, y2 = bbox

                        # 选择颜色
                        if current_prediction:
                            color = color_map.get(current_prediction["class"], (128, 128, 128))
                            detection_text = (
                                f"ID{track_id}: {current_prediction['class']}: {current_prediction['confidence']:.2f}"
                            )
                        else:
                            color = (128, 128, 128)
                            detection_text = f"ID{track_id}: Person: {conf:.2f}"

                        # 绘制检测框和文字
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(
                            annotated_frame, detection_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )

                    # 在右上角显示检测统计
                    if filtered_boxes:
                        stats_text = f"Targets: {len(filtered_boxes)}"
                        cv2.putText(
                            annotated_frame,
                            stats_text,
                            (width - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                    # 绘制信息背景
                    cv2.rectangle(annotated_frame, (10, 10), (width - 10, 120), (0, 0, 0), -1)
                    cv2.rectangle(annotated_frame, (10, 10), (width - 10, 120), (255, 255, 255), 2)

                    # 绘制时间和预测信息
                    time_text = f"Time: {current_time}s / {video_info.get('duration', 0):.0f}s"
                    cv2.putText(annotated_frame, time_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    if current_prediction:
                        pred_text = f"Behavior: {current_prediction['class']}"
                        conf_text = f"Confidence: {current_prediction['confidence']:.3f}"
                        segment_text = f"Segment: {current_prediction['segment_info']}"

                        text_color = color_map.get(current_prediction["class"], (255, 255, 255))
                        cv2.putText(annotated_frame, pred_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                        cv2.putText(
                            annotated_frame, conf_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                        )
                        cv2.putText(
                            annotated_frame, segment_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
                        )
                    else:
                        cv2.putText(
                            annotated_frame,
                            "Behavior: No prediction",
                            (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (128, 128, 128),
                            2,
                        )

                    out.write(annotated_frame)
                    frame_count += 1
                    pbar.update(1)

        finally:
            cap.release()
            out.release()

        print(f"✅ Sequence visualization video saved to: {output_path}")
        return output_path


def main():
    """主函数 - 简化的命令行接口"""
    parser = argparse.ArgumentParser(description="Video Sequence Behavior Prediction")

    # 必需参数
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--output_dir", type=str, default="output/sequence", help="Output directory")

    # 模型参数
    parser.add_argument(
        "--model_path", type=str, default="checkpoints/yolodetect/best_model.pth", help="Model checkpoint path"
    )
    parser.add_argument(
        "--reference_embeddings",
        type=str,
        default="checkpoints/yolodetect/val_embeddings.pt",
        help="Reference embeddings path",
    )
    parser.add_argument("--yolo_model_path", type=str, default="weights/yolov11n.pt", help="YOLO model path")

    # 序列预测参数
    parser.add_argument("--segment_duration", type=float, default=3.0, help="Segment duration in seconds")
    parser.add_argument("--overlap_ratio", type=float, default=0.3, help="Overlap ratio between segments")
    parser.add_argument("--frames_per_segment", type=int, default=30, help="Frames per segment")
    parser.add_argument("--confidence_threshold", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--max_segments", type=int, default=None, help="Max segments for testing")

    # 可视化参数
    parser.add_argument("--visualize", action="store_true", help="Create visualization video")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化预测器
    print("🚀 Initializing Sequence Video Predictor...")
    predictor = SequenceVideoPredictor(
        model_path=args.model_path,
        reference_embeddings_path=args.reference_embeddings,
        yolo_model_path=args.yolo_model_path,
    )

    # 进行序列预测
    print(f"🎯 Starting sequence prediction on: {args.video_path}")
    result = predictor.predict_sequence(
        video_path=args.video_path,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap_ratio,
        frames_per_segment=args.frames_per_segment,
        confidence_threshold=args.confidence_threshold,
        max_segments=args.max_segments,
    )

    # 打印结果摘要
    summary = result["summary"]
    print("\n📊 Prediction Summary:")
    print(f"Dominant behavior: {summary['dominant_behavior']}")
    print(f"Confident predictions: {summary['total_confident_predictions']}/{summary['total_segments']}")
    print(f"Coverage ratio: {summary['coverage_ratio']:.1%}")
    print(f"Behavior distribution: {summary['behavior_distribution']}")
    print(f"Average confidence: {summary['average_confidence']:.3f}")

    # 保存结果
    video_name = Path(args.video_path).stem
    json_output_path = os.path.join(args.output_dir, f"{video_name}_sequence_prediction.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"📄 Results saved to: {json_output_path}")

    # 生成可视化视频
    if args.visualize:
        video_output_path = os.path.join(args.output_dir, f"{video_name}_sequence_annotated.mp4")
        predictor.create_visualization_video(
            video_path=args.video_path, sequence_results=result, output_path=video_output_path
        )
        print(f"🎬 Visualization video saved to: {video_output_path}")

    print("✅ Sequence prediction completed!")


if __name__ == "__main__":
    main()


"""
使用示例:

# python predict.py --video_path data/test/merged_final.mp4  --visualize
"""
