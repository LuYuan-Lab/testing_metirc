# pre_crop_videos.py
#方便训练前预处理数据集，生成裁剪框 目标的JSON 文件
import os
import json
from tool.auto_crop import AutoCropper

def generate_crop_boxes(data_root: str, output_json: str):
    """
    遍历整个数据集，为每个视频检测人框并保存到 JSON。
    """
    cropper = AutoCropper(
        model_path="yolov11n.pt",
        conf_thres=0.5,
        target_class="person",
        margin_ratio=0.1
    )

    crop_dict = {}

    # 遍历 train/val 等所有子文件夹
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, f)
                try:
                    crop_rect = cropper.detect_video_crop(video_path)
                    crop_dict[video_path] = crop_rect
                    print(f"✅ {video_path} -> {crop_rect}")
                except Exception as e:
                    print(f"⚠️ Failed to process {video_path}: {e}")

    # 保存到 JSON 文件
    with open(output_json, 'w') as fp:
        json.dump(crop_dict, fp, indent=4)
    print(f"\n🎯 Saved {len(crop_dict)} crop boxes to {output_json}")

if __name__ == "__main__":
    data_root = "data"  # 你的数据根目录
    output_json = "boxes_json/crop_boxes.json"
    generate_crop_boxes(data_root, output_json)
