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
│   ├── best_model.pth        # 训练好的最佳模型
│   └── val_tsne_plot.png     # 验证集t-SNE可视化结果
├── dataset.py                # 数据集定义和预处理
├── loss.py                   # 三元组损失函数封装
├── model.py                  # R(2+1)D模型定义
├── train.py                  # 模型训练脚本
├── test.py                   # 模型评估脚本
└── README.md                 # 本文档
```

## ⚙️ 环境搭建

1.  **克隆项目** (如果需要)
    ```bash
    git clone <your-repo-url>
    cd testing_metirc
    ```

2.  **创建虚拟环境** (推荐)
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **安装依赖**
    项目依赖以下库，您可以使用 `pip` 进行安装：
    ```bash
    pip install torch torchvision
    pip install opencv-python numpy scikit-learn matplotlib tqdm pytorch-metric-learning Pillow
    ```
    *建议在一个 `requirements.txt` 文件中管理这些依赖。*

## 准备数据集

将您的视频数据集按以下结构组织在 `data` 目录下：

-   每个子目录代表一个行为类别（如 `举手`, `正常`）。
-   训练视频放入 `data/train/` 下对应的类别文件夹。
-   验证视频放入 `data/val/` 下对应的类别文件夹。

视频格式支持 `.mp4`, `.avi`, `.mov` 等 OpenCV 可读取的格式。

## 🚀 如何使用

### 1. 训练模型

通过运行 `train.py` 脚本来开始训练。您可以自定义训练参数。

**基本命令:**
```bash
python train.py
```

**常用参数:**
- `--data_root`: 数据集根目录 (默认: `data`)。
- `--output_dir`: 模型检查点保存目录 (默认: `checkpoints`)。
- `--epochs`: 训练轮数 (默认: `30`)。
- `--lr`: 学习率 (默认: `1e-4`)。
- `--p_classes`: 每个批次中包含的类别数 (P) (默认: `5`)。
- `--k_samples`: 每个类别在批次中的样本数 (K) (默认: `4`)。
- `--embedding_dim`: 输出特征向量的维度 (默认: `128`)。
- `--num_frames`: 从每个视频中采样的帧数 (默认: `30`)。
- `--freeze_layers`: 需要冻结的模型层列表 (例如 `['stem', 'layer1']`)。

**示例 (使用自定义参数):**
```bash
python train.py --epochs 50 --lr 1e-5 --p_classes 5 --k_samples 8
```

训练过程中，验证损失最低的模型将被保存为 `checkpoints/best_model.pth`。

### 2. 评估模型

训练完成后，使用 `test.py` 脚本来评估模型的性能。该脚本会执行以下操作：
1.  从训练集和验证集中提取所有视频的特征向量。
2.  使用 `k-NN` 分类器在验证集上进行分类，并输出准确率和分类报告。
3.  对验证集的特征向量进行 `t-SNE` 降维，并生成可视化图像 `val_tsne_plot.png`，保存在模型目录中。

**运行命令:**
```bash
python test.py --model_path checkpoints/best_model.pth
```

**参数说明:**
- `--model_path`: 指向训练好的模型权重文件。
- `--batch_size`: 特征提取时的批处理大小 (可以设置得比训练时大) (默认: `32`)。

评估结果将直接打印在控制台，`t-SNE` 图像将帮助您直观地判断不同类别的视频特征是否在空间上被有效分离开。
