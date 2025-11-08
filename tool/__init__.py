"""
工具模块初始化文件
"""

# 训练阶段工具
from .dataset import VideoDataset
from .loss import TripletLossWrapper

# 推理阶段工具
from .tracker import TrackingConfig, UltralyticsTracker, VideoTracker
from .video_crop_processor import AutoCropper, VideoCropProcessor

__all__ = [
    # 训练阶段
    "VideoDataset",
    "TripletLossWrapper",
    "AutoCropper",
    "VideoCropProcessor",
    # 推理阶段
    "VideoTracker",
    "UltralyticsTracker",
    "TrackingConfig",
]
