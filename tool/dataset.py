import os
import cv2
import torch
import random
import json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from typing import Tuple, List
from PIL import Image
from tool.auto_crop import AutoCropper


class VideoDataset(Dataset):

    def __init__(self, 
                 data_root: str, 
                 mode: str = 'train', 
                 num_frames: int = 30,
                 random_offset: int = 50,
                 resize_shape: Tuple[int, int] = (112, 112),
                 crop_json: str = "boxes_json/crop_boxes.json"):

        """
        Args:
            data_root (str): 数据集根目录
            mode (str): 'train' or 'val'
            num_frames (int): 采样帧数
            random_offset (int): 训练时裁剪随机偏移
            resize_shape (Tuple[int,int]): 输出图像大小
            crop_json (str): 存储预生成裁剪框的 JSON 文件路径
        """
        assert mode in ['train', 'val']
        
        self.data_path = os.path.join(data_root, mode)
        self.mode = mode
        self.num_frames = num_frames
        self.random_offset = random_offset
        self.resize_shape = resize_shape

        # ✅ 加载 YOLO 备用（仅在 JSON 缺失某视频时调用）
        self.auto_cropper = AutoCropper(
            model_path="weights/yolov11n.pt",
            conf_thres=0.5,
            target_class="person",
            margin_ratio=0.1
        )

        # ✅ 加载裁剪框缓存
        if os.path.exists(crop_json):
            with open(crop_json, "r") as f:
                self.crop_cache = json.load(f)
            print(f"✅ Loaded {len(self.crop_cache)} cached crop boxes from {crop_json}")
        else:
            print(f"⚠️ crop_boxes.json not found, YOLO will run on the fly.")
            self.crop_cache = {}

        self.video_files = []
        self.class_to_idx = {}
        
        self._find_classes_and_videos()
        self._build_transforms()


    def _find_classes_and_videos(self):
        """扫描类别文件夹"""
        classes = sorted(entry.name for entry in os.scandir(self.data_path) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"No class folders found in {self.data_path}")
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(self.data_path, class_name)
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.video_files.append((os.path.join(class_path, video_file), class_idx))
        
        print(f"✅ Found {len(self.video_files)} videos in {len(classes)} classes for '{self.mode}' mode.")


    def _build_transforms(self):
        """构建图像变换流水线"""
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.pil_transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.video_files)


    def __getitem__(self, idx: int):
        video_path, label = self.video_files[idx]
        frames = self._sample_and_crop_frames(video_path)

        if not frames:
            print(f"⚠️ Failed to load video {video_path}, skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # 随机增强（只对train）
        if self.mode == 'train':
            do_flip = random.random() < 0.2
            brightness = random.uniform(0.8, 1.2) if random.random() < 0.5 else 1.0
            contrast = random.uniform(0.8, 1.2) if random.random() < 0.5 else 1.0
            saturation = random.uniform(0.8, 1.2) if random.random() < 0.5 else 1.0

            pil_frames = [Image.fromarray(f) for f in frames]
            if do_flip:
                pil_frames = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in pil_frames]
            pil_frames = [F.adjust_brightness(f, brightness) for f in pil_frames]
            pil_frames = [F.adjust_contrast(f, contrast) for f in pil_frames]
            pil_frames = [F.adjust_saturation(f, saturation) for f in pil_frames]
            processed = torch.stack([self.pil_transform(f) for f in pil_frames])
        else:
            processed = torch.stack([self.base_transform(f) for f in frames])

        # (T, C, H, W) -> (C, T, H, W)
        return processed.permute(1, 0, 2, 3), label


    def _sample_and_crop_frames(self, video_path: str) -> List[np.ndarray]:
        """采样并裁剪帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        # ✅ 读取缓存框（或重新检测）
        crop_rect = self.crop_cache.get(video_path)
        if crop_rect is None:
            crop_rect = self.auto_cropper.detect_video_crop(video_path)
            self.crop_cache[video_path] = crop_rect  # 可选：更新缓存
        
        x1, y1, x2, y2 = map(int, crop_rect)
        if self.mode == 'train':
            x1 += random.randint(-self.random_offset, self.random_offset)
            y1 += random.randint(-self.random_offset, self.random_offset)
            x2 += random.randint(-self.random_offset, self.random_offset)
            y2 += random.randint(-self.random_offset, self.random_offset)

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            cropped = frame[y1:y2, x1:x2]
            frames.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames
