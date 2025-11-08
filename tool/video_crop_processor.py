"""
视频裁剪处理器
整合了自动裁剪检测和批量视频处理功能
"""

import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class AutoCropper:
    """
    使用 YOLO 检测目标区域并返回裁剪框，只保留指定类别（如 'person'）。
    """

    def __init__(
        self,
        model_path: str = "weights/yolov11n.pt",
        conf_thres: float = 0.5,
        target_class=None,
        margin_ratio: float = 0.1,
        default_crop_rect: Tuple[int, int, int, int] = (720, 120, 1700, 1000),
        max_missing_frames: int = 30,
    ):
        """
        Args:
            model_path: YOLO 模型路径或模型名
            conf_thres: 检测置信度阈值
            target_class: 想要检测的类别，可以是类别ID(int) 或 名称(str)，如 'person'
            margin_ratio: 扩大 bbox 的比例
            default_crop_rect: 检测失败时使用的默认裁剪区域
            max_missing_frames: 连续多少帧未检测到目标后，判定为真正丢失
        """
        try:
            self.model = YOLO(model_path)
        except FileNotFoundError:
            print(f"⚠️ 模型 {model_path} 未找到，尝试使用 YOLOv8n 替代")
            self.model = YOLO("yolov8n.pt")

        self.model.fuse()
        self.model.conf = conf_thres
        self.target_class = target_class
        self.margin_ratio = margin_ratio
        self.default_crop_rect = default_crop_rect
        self.max_missing_frames = max_missing_frames

        # 状态缓存
        self.last_crop_rect: Optional[Tuple[int, int, int, int]] = None
        self.missing_count: int = 0

        # ✅ 映射类别名到 ID
        if isinstance(self.target_class, str):
            names = self.model.names
            name_to_id = {v: k for k, v in names.items()}
            if self.target_class in name_to_id:
                mapped_id = name_to_id[self.target_class]
                print(f"✅ 检测目标 '{self.target_class}' -> 类别ID {mapped_id}")
                self.target_class = mapped_id
            else:
                print(f"⚠️ 未找到类别名 '{self.target_class}'，将检测所有类别。")
                self.target_class = None

    def detect_crop_rect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        输入一帧 RGB 图像，返回裁剪框 (x1, y1, x2, y2)
        若连续超过 max_missing_frames 帧未检测到目标，则返回 None
        """
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            self.missing_count += 1
            if self.last_crop_rect is not None and self.missing_count < self.max_missing_frames:
                return self.last_crop_rect
            else:
                return None

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy_all = boxes.xyxy.cpu().numpy()

        # 只保留目标类别
        if self.target_class is not None:
            mask = cls_ids == self.target_class
            if np.any(mask):
                xyxy_all = xyxy_all[mask]
            else:
                self.missing_count += 1
                if self.last_crop_rect is not None and self.missing_count < self.max_missing_frames:
                    return self.last_crop_rect
                else:
                    return None

        # 找面积最大目标
        areas = (xyxy_all[:, 2] - xyxy_all[:, 0]) * (xyxy_all[:, 3] - xyxy_all[:, 1])
        largest_idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, xyxy_all[largest_idx])

        # 添加 margin
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w * self.margin_ratio))
        y1 = max(0, int(y1 - h * self.margin_ratio))
        x2 = int(x2 + w * self.margin_ratio)
        y2 = int(y2 + h * self.margin_ratio)

        crop_rect = (x1, y1, x2, y2)

        # 更新状态
        self.last_crop_rect = crop_rect
        self.missing_count = 0
        return crop_rect

    def detect_video_crop(self, video_path: str) -> Tuple[int, int, int, int]:
        """
        对视频前几帧检测目标，返回统一裁剪框
        若所有帧都未检测到目标，则返回 default_crop_rect
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ 无法打开视频 {video_path}，使用默认裁剪框")
            return self.default_crop_rect

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(5, total_frames)

        crop_rects = []
        for idx in range(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rect = self.detect_crop_rect(rgb_frame)
            if rect is not None:
                crop_rects.append(rect)

        cap.release()

        if not crop_rects:
            return self.default_crop_rect

        # 合并有效检测框
        x1 = min([r[0] for r in crop_rects])
        y1 = min([r[1] for r in crop_rects])
        x2 = max([r[2] for r in crop_rects])
        y2 = max([r[3] for r in crop_rects])

        return x1, y1, x2, y2


class VideoCropProcessor:
    """
    批量视频裁剪处理器
    """
    
    def __init__(self, 
                 model_path: str = "weights/yolov11n.pt",
                 conf_thres: float = 0.5,
                 target_class: str = "person",
                 margin_ratio: float = 0.1):
        """
        初始化视频裁剪处理器
        
        Args:
            model_path: YOLO 模型路径
            conf_thres: 检测置信度阈值
            target_class: 检测目标类别
            margin_ratio: 裁剪框扩展比例
        """
        self.cropper = AutoCropper(
            model_path=model_path,
            conf_thres=conf_thres,
            target_class=target_class,
            margin_ratio=margin_ratio,
        )
    
    def generate_crop_boxes(self, data_root: str, output_json: str):
        """
        遍历整个数据集，为每个视频检测人框并保存到 JSON。
        
        Args:
            data_root: 数据集根目录
            output_json: 输出JSON文件路径
        """
        crop_dict = {}

        # 遍历 train/val 等所有子文件夹
        for root, _, files in os.walk(data_root):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_path = os.path.join(root, f)
                    try:
                        crop_rect = self.cropper.detect_video_crop(video_path)
                        crop_dict[video_path] = crop_rect
                        print(f"✅ {video_path} -> {crop_rect}")
                    except Exception as e:
                        print(f"⚠️ Failed to process {video_path}: {e}")

        # 保存到 JSON 文件
        output_dir = os.path.dirname(output_json)
        if output_dir:  # 只有当目录路径不为空时才创建目录
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, "w") as fp:
            json.dump(crop_dict, fp, indent=4)
        print(f"\n🎯 Saved {len(crop_dict)} crop boxes to {output_json}")
        return crop_dict
    
    def process_single_video(self, video_path: str) -> Tuple[int, int, int, int]:
        """
        处理单个视频，返回裁剪框
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            裁剪框坐标 (x1, y1, x2, y2)
        """
        return self.cropper.detect_video_crop(video_path)
    
    def load_crop_boxes(self, json_path: str) -> dict:
        """
        从JSON文件加载裁剪框数据
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            包含视频路径和裁剪框的字典
        """
        try:
            with open(json_path, "r") as fp:
                crop_dict = json.load(fp)
            print(f"✅ 从 {json_path} 加载了 {len(crop_dict)} 个裁剪框")
            return crop_dict
        except FileNotFoundError:
            print(f"⚠️ 文件 {json_path} 不存在")
            return {}
        except json.JSONDecodeError:
            print(f"⚠️ JSON 文件 {json_path} 格式错误")
            return {}


def main():
    """
    主函数 - 批量处理数据集中的所有视频
    """
    # 配置参数
    data_root = "data"  # 你的数据根目录
    output_json = "boxes_json/crop_boxes.json"
    
    # 创建处理器
    processor = VideoCropProcessor(
        model_path="weights/yolov11n.pt",
        conf_thres=0.5,
        target_class="person",
        margin_ratio=0.1
    )
    
    # 生成裁剪框
    crop_boxes = processor.generate_crop_boxes(data_root, output_json)
    
    print(f"\n📊 处理完成！共处理 {len(crop_boxes)} 个视频文件")


if __name__ == "__main__":
    main()
