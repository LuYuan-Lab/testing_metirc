# 考场行为识别与监控AI算法

本项目是一个基于深度学习的视频理解项目，旨在通过度量学习（Metric Learning）训练一个能够区分不同考场行为（如举手、玩手机、站立等）的AI模型。模型利用3D卷积神经网络（R(2+1)D）从视频片段中提取高质量的特征向量（Embeddings），使得相似行为的视频在特征空间中距离更近，不同行为的视频距离更远。

## ✨ 项目特点

- **先进的模型结构**: 采用 `R(2+1)D` 视频识别模型，能有效捕捉时空特征。
- **度量学习**: 使用三元组损失（Triplet Loss）和 `P-K` 采样策略进行端到端的度量学习，专注于学习具有区分性的特征。
- **高效的数据处理**: 自定义的 `VideoDataset` 类，支持视频帧采样、特定区域裁剪、数据增强（随机偏移、翻转、颜色扰动）等。
- **全面的模型评估**: 提供 `k-NN` 分类准确率评估和 `t-SNE` 可视化，直观地衡量模型学习到的特征质量。
- **模块化设计**: 代码结构清晰，分为数据处理、模型定义、损失函数、训练和测试脚本，易于理解和扩展。

## 📂 项目结构

```
.
├── data/                     # 数据集根目录
│   ├── train/                # 训练集
│   │   ├── 举手/
│   │   ├── 手机/
│   │   └── ...
│   └── val/                  # 验证集
│       ├── 举手/
│       └── ...
├── checkpoints/              # 模型权重和输出文件
# 考场行为识别与监控 — 项目说明

这是一个用于考场行为识别的度量学习项目。目标是使用视频片段训练一个能够提取判别性 Embedding 的模型，便于后续使用 k-NN 或其他检索/分类方法对行为进行识别。

核心思想：使用 R(2+1)D 风格的时空卷积网络提取视频特征，并通过 Triplet Loss（配合 P-K 采样）进行度量学习，使得相同行为的视频在特征空间距离更近。

## 主要特点

- 使用 R(2+1)D / VideoResNet 风格的视频骨干网络（项目内含 `model/ResNetModel.py` 的自定义实现以及 `model/model.py` 的 torchvision 接口）。
- 度量学习训练：Triplet Loss + TripletMarginMiner（semihard 等挖掘器），并使用自定义的 P-K Batch Sampler 保证每个 batch 中包含 P 类、每类 K 个样本。
- 自动裁剪辅助：支持基于 YOLO 的视频区域自动裁剪（`tool/auto_crop.py` 与 `scripts/batch_crop.py`）。
- 评估：提取 Embeddings 后使用 k-NN 评估，并提供 t-SNE 可视化（`test.py`）。

## 项目结构与文件说明

```
.
├── data/                      # 用户的数据根目录（train/val 子目录）
├── checkpoints/               # 模型和评估结果（训练过程会保存 best_model.pth）
├── model/                     # 两个 model 实现：torchvision 接口和自定义 ResNet 视频实现
│   ├── model.py               # 基于 torchvision r2plus1d_18 的工厂函数 R2Dmodel
│   ├── ResNetModel.py         # 自定义 VideoResNet / Conv2Plus1D 实现与 R2Dmodel
│   └── CustomModel2.py        # 备用/实验型模型（可选）
├── tool/                      # 工具集合
│   ├── dataset.py             # `VideoDataset`：视频读取、采样、裁剪与增强
│   ├── loss.py                # `TripletLossWrapper`：Triplet 损失 + miner
│   ├── auto_crop.py           # 基于 ultralytics YOLO 的自动裁剪工具（返回裁剪框或 None）
│   ├── view_cropping.py       # 可视化裁剪结果的脚本（查看样例裁剪）
│   └── per_crop_videos.py     # 按视频生成裁剪图像的辅助脚本（实验性）
├── scripts/
│   └── batch_crop.py          # 使用 auto_crop 批量裁剪并保存视频帧为 JPEG
├── train.py                   # 训练入口：包含 PKBatchSampler、训练/验证循环
├── test.py                    # 评估入口：特征提取、k-NN 评估、t-SNE 可视化
└── README.md                  # 本文档（当前文件）
```

## 依赖与环境

建议使用 conda 或 venv 创建隔离环境，安装主要依赖：

```bash
pip install torch torchvision
pip install opencv-python numpy matplotlib scikit-learn tqdm pillow
pip install pytorch-metric-learning
# 若需要自动裁剪功能：
pip install ultralytics
```

建议将实际依赖导出为 `requirements.txt` 并提交，以保证可复现性。

## 快速开始

1) 准备数据：

        data/
        ├── train/
        │   ├── class1/  # 每个行为一个文件夹，里面放视频
        │   └── class2/
        └── val/

2) 训练模型（默认配置）：

```bash
python train.py --data_root data --output_dir checkpoints/yolodetect --epochs 30
```

常用参数说明（`train.py`）:

- `--data_root`: 数据根路径（默认 `data`）
- `--output_dir`: 保存模型的目录（默认 `checkpoints/yolodetect`）
- `--p_classes` / `--k_samples`: P-K 采样参数
- `--embedding_dim`: 输出 Embedding 维度
- `--num_frames`: 每个视频采样帧数

训练期间会保存验证集上表现最好的模型到 `output_dir/best_model.pth`。

3) 评估模型：

```bash
python test.py --model_path checkpoints/yolodetect/best_model.pth --data_root data
```

`test.py` 会输出 k-NN 的准确率和分类报告，并在模型目录生成 `val_tsne_plot.png` 和 `val_embeddings.pt`。

## 自动裁剪（可选）

如果数据中包含大画面或场景需要先裁剪到考生区域，可使用项目自带的自动裁剪工具：

- `tool/auto_crop.py`：基于 ultralytics YOLO 的 `AutoCropper`。关键行为：
    - 单帧检测失败时，会使用上一帧的检测框作为回退。
    - 若连续 `max_miss_frames`（默认 30）帧未检测到目标，缓存会被清空，后续返回 `None`。
    - `detect_video_crop(video_path)` 会在若干采样帧上检测并返回所有有效检测框的并集，或在无有效检测时返回 `default_crop_rect`（或 None，取决于配置）。

- `scripts/batch_crop.py`：遍历输入目录并将裁剪后的帧保存为 JPEG；当 `detect_video_crop` 返回 `None` 时脚本会跳过该视频。

示例命令：

```bash
python scripts/batch_crop.py --input_dir data/train --output_dir data_cropped/train --weights weights/yolov11n.pt
```

注意：如果本地没有 YOLO 权重，`ultralytics` 可能尝试自动下载模型（需要网络）；也可自行放置权重到 `weights/`。

## 代码要点与可配置项

- `tool/dataset.py` (`VideoDataset`)：
    - 使用 `AutoCropper` 获取每个视频的裁剪框（支持从 `crop_boxes.json` 加载缓存以加速）。
    - 采样 `num_frames` 帧并返回形状为 `(C, T, H, W)` 的张量供模型使用。
    - 训练模式下会进行简单数据增强（翻转、亮度/对比度/饱和度扰动）。

- `train.py`：
    - 包含 `PKBatchSampler` 实现 P-K 采样。
    - 使用 `TripletLossWrapper`（`tool/loss.py`）计算损失并使用 Adam 优化器训练。

- `model/ResNetModel.py` 与 `model/model.py`：
    - 两种 R(2+1)D 实现：自定义 VideoResNet（`ResNetModel.py`）和 torchvision 接口（`model.py`）。
    - 通过 `R2Dmodel(embedding_dim, pretrained, freeze_layers)` 获取用于度量学习的 embedding 网络。



