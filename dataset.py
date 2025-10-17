import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import av
import random

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        """
        Args:
            root_dir (string): 包含 'train' 或 'val' 文件夹的目录。
            num_frames (int): 从每个视频中采样的帧数。
            transform (callable, optional): 应用于每帧的转换。
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        self.samples = self._make_dataset(self.root_dir, self.class_to_idx)
        
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, directory, class_to_idx):
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_index)
                    instances.append(item)
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        frames = []
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            
            # 处理视频帧数不足的情况
            if total_frames == 0:
                print(f"警告: 视频 {video_path} 没有帧")
                total_frames = 1  # 设置为1以避免除零错误
            
            # 读取所有帧
            all_frames = []
            for frame in container.decode(video=0):
                img = frame.to_image()
                all_frames.append(img)
            
            # 如果实际帧数与声明的不一致，使用实际帧数
            actual_frames = len(all_frames)
            if actual_frames == 0:
                print(f"警告: 无法从视频 {video_path} 读取任何帧")
                # 创建黑色默认帧
                from PIL import Image
                default_img = Image.new('RGB', (224, 224), color='black')
                all_frames = [default_img]
                actual_frames = 1
            
            # 根据实际帧数和需要的帧数进行处理
            if actual_frames >= self.num_frames:
                # 如果帧数足够，均匀采样
                indices = torch.linspace(0, actual_frames - 1, self.num_frames, dtype=torch.long)
                frames = [all_frames[i] for i in indices]
            else:
                # 如果帧数不够，重复帧来填充
                print(f"警告: 视频 {video_path} 只有 {actual_frames} 帧，需要 {self.num_frames} 帧，将重复帧")
                frames = all_frames.copy()
                
                # 通过重复最后一帧来填充
                while len(frames) < self.num_frames:
                    frames.append(all_frames[-1])  # 重复最后一帧
                
                # 如果仍然超出，截取到需要的帧数
                frames = frames[:self.num_frames]

        except Exception as e:
            print(f"读取视频失败: {video_path}, 错误: {e}")
            # 如果视频损坏，创建默认的图像并应用变换以保持一致性
            from PIL import Image
            default_img = Image.new('RGB', (224, 224), color='black')
            frames = [default_img] * self.num_frames

        # 确保我们有正确数量的帧
        if len(frames) != self.num_frames:
            print(f"警告: 帧数不匹配，期望 {self.num_frames}，实际 {len(frames)}")
            # 强制调整到正确的帧数
            if len(frames) < self.num_frames:
                # 填充不足的帧
                last_frame = frames[-1] if frames else Image.new('RGB', (224, 224), color='black')
                frames.extend([last_frame] * (self.num_frames - len(frames)))
            else:
                # 截取多余的帧
                frames = frames[:self.num_frames]

        # 应用图像变换
        if self.transform:
            frames = [self.transform(img) for img in frames]
            
        # 将帧列表堆叠成一个张量 (T, C, H, W) -> (C, T, H, W)
        # T=num_frames, C=3
        if frames:
            video_tensor = torch.stack(frames, dim=1) # (C, T, H, W)
        else:
            # 以防万一没采到帧，创建默认图像并应用变换
            from PIL import Image
            default_img = Image.new('RGB', (224, 224), color='black')
            default_frames = [default_img] * self.num_frames
            
            if self.transform:
                default_frames = [self.transform(img) for img in default_frames]
            
            video_tensor = torch.stack(default_frames, dim=1)

        return video_tensor, label