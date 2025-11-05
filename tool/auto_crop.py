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
