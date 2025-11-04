#查看裁剪检测框

from tool.dataset import VideoDataset

# 数据集根目录
data_root = "data"  # 改成你实际的视频数据目录
mode = "train"

# 初始化 VideoDataset
dataset = VideoDataset(
    data_root=data_root,
    mode=mode,
    num_frames=10,  # 测试时可以只采样少量帧，加快速度
    random_offset=0,  # 测试时先不要随机偏移
    resize_shape=(112, 112)
)

print(f"Found {len(dataset)} videos in dataset.")

# 遍历前几个视频，打印裁剪框
for idx in range(min(5, len(dataset))):
    video_path, label = dataset.video_files[idx]
    crop_rect = dataset.auto_cropper.detect_video_crop(video_path)
    print(f"Video: {video_path}")
    print(f"Label: {label}, Crop Rect (x1, y1, x2, y2): {crop_rect}")

    # 如果想看裁剪效果，可以读取第一帧并裁剪后显示
    import cv2
    import matplotlib.pyplot as plt

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        x1, y1, x2, y2 = crop_rect
        cropped = frame[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        plt.imshow(cropped_rgb)
        plt.title(f"Video {idx} - Cropped Frame")
        plt.axis("off")
        plt.show()
    cap.release()
