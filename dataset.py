import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from typing import Tuple, List
from PIL import Image

class VideoDataset(Dataset):

    def __init__(self, 
                 data_root: str, 
                 mode: str = 'train', 
                 num_frames: int = 30,
                 crop_rect: Tuple[int, int, int, int] = (720, 120, 1700, 1000),
                 random_offset: int = 50,
                 resize_shape: Tuple[int, int] = (112, 112)):
        """
        初始化数据集。

        Args:
            data_root (str): 数据集的根目录 (例如 'data/')。
            mode (str): 'train' 或 'val'，决定加载哪个数据集并是否应用数据增强。
            num_frames (int): 从每个视频中均匀采样的帧数。
            crop_rect (Tuple[int, int, int, int]): 裁剪区域的左上角和右下角坐标 (x1, y1, x2, y2)。
            random_offset (int): 训练时，在裁剪坐标上应用的最大随机偏移像素。
            resize_shape (Tuple[int, int]): 裁剪后将帧调整到的大小 (height, width)。
        """
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'."
        
        self.data_path = os.path.join(data_root, mode)
        self.mode = mode
        self.num_frames = num_frames
        self.crop_rect = crop_rect
        self.random_offset = random_offset
        self.resize_shape = resize_shape

        self.video_files = []
        self.class_to_idx = {}
        
        self._find_classes_and_videos()
        self._build_transforms()

    def _find_classes_and_videos(self):
        """扫描目录，找到所有类别和对应的视频文件。"""
        classes = sorted(entry.name for entry in os.scandir(self.data_path) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {self.data_path}")
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(self.data_path, class_name)
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.video_files.append((os.path.join(class_path, video_file), class_idx))
        
        print(f"Found {len(self.video_files)} videos in {len(classes)} classes for '{self.mode}' mode.")

    def _build_transforms(self):
        """构建图像变换流水线。"""
        # 包含ToPILImage的完整变换（用于numpy数组）
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 不包含ToPILImage的变换（用于已经是PIL的图像）
        self.pil_transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个数据样本。

        Returns:
            torch.Tensor: 处理后的视频帧张量，形状为 (C, T, H, W)。
            int: 视频的类别标签。
        """
        video_path, label = self.video_files[idx]

        # 1. 采样并裁剪帧
        frames = self._sample_and_crop_frames(video_path)
        if not frames:
            print(f"Warning: Failed to load video {video_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # 2. 采样一次随机参数，对所有帧应用同样的变换
        if self.mode == 'train':
            # 随机翻转
            do_flip = random.random() < 0.2
            # 随机颜色扰动参数（采样一次参数）
            brightness_factor = random.uniform(0.8, 1.2) if random.random() < 0.5 else 1.0
            contrast_factor = random.uniform(0.8, 1.2) if random.random() < 0.5 else 1.0
            saturation_factor = random.uniform(0.8, 1.2) if random.random() < 0.5 else 1.0
            
            # 采样一次参数
            pil_frames = [Image.fromarray(frame) for frame in frames]
            if do_flip:
                pil_frames = [frame.transpose(Image.FLIP_LEFT_RIGHT) for frame in pil_frames]
            
            # 对所有帧应用相同的颜色扰动参数
            pil_frames = [F.adjust_brightness(frame, brightness_factor) for frame in pil_frames]
            pil_frames = [F.adjust_contrast(frame, contrast_factor) for frame in pil_frames]
            pil_frames = [F.adjust_saturation(frame, saturation_factor) for frame in pil_frames]
            
            processed_frames = torch.stack([self.pil_transform(frame) for frame in pil_frames])
        else:
            processed_frames = torch.stack([self.base_transform(frame) for frame in frames])

        # 3. 调整维度顺序以匹配视频模型输入 (C, T, H, W)
        processed_frames = processed_frames.permute(1, 0, 2, 3)

        return processed_frames, label

    def _sample_and_crop_frames(self, video_path: str) -> List[np.ndarray]:
        """从视频文件中均匀采样、裁剪帧。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            print(f"Warning: Video {video_path} has only {total_frames} frames, less than required {self.num_frames}. Skipping.")
            cap.release()
            return []

        # 计算采样帧的索引
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        
        # 计算本次裁剪的坐标
        x1, y1, x2, y2 = self.crop_rect
        if self.mode == 'train':
            offset_x = random.randint(-self.random_offset, self.random_offset)
            offset_y = random.randint(-self.random_offset, self.random_offset)
            x1, x2 = x1 + offset_x, x2 + offset_x
            y1, y2 = y1 + offset_y, y2 + offset_y
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 裁剪
                cropped_frame = frame[y1:y2, x1:x2]
                # OpenCV 读取的是 BGR, 转换为 RGB
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            else:
                # 如果某一帧读取失败，可以忽略
                continue
        
        cap.release()

        # 确保我们得到了足够数量的帧
        if len(frames) < self.num_frames:
             print(f"Warning: Only able to read {len(frames)}/{self.num_frames} frames from {video_path}.")
             return []
             
        return frames
