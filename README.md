# 考场行为识别与监控 AI 系统

基于深度学习和度量学习的视频理解系统，用于识别考场中的异常行为。采用 R(2+1)D 视频网络提取时空特征，通过 Triplet Loss 和 P-K 采样策略训练判别性特征表示。

## 🎯 核心特性

- **时空特征提取**：R(2+1)D 视频识别模型，有效捕捉时空特征
- **度量学习训练**：Triplet Loss + Semi-hard/Hard 挖掘策略
- **智能批次采样**：P-K 采样确保有效的三元组构建
- **YOLO 自动裁剪**：基于目标检测的视频区域智能裁剪
- **全面评估体系**：k-NN 分类 + t-SNE 可视化 + 完整测试套件

## 📁 项目结构

```
.
├── model/                    # 模型实现
│   ├── ResNetModel.py       # 自定义 R(2+1)D 视频网络
│   ├── model.py            # torchvision 接口封装
│   └── CustomModel2.py     # 实验性模型
├── tool/                    # 核心工具集
│   ├── dataset.py         # 视频数据集与预处理
│   ├── loss.py           # Triplet Loss 封装
│   ├── auto_crop.py      # YOLO 自动裁剪
│   ├── view_cropping.py  # 裁剪可视化工具
│   └── per_crop_videos.py # 视频裁剪脚本
├── tests/                  # 测试套件
│   ├── test_dataset.py   # 数据集测试
│   ├── test_model.py     # 模型测试
│   ├── test_loss.py      # 损失函数测试
│   └── test_sampler.py   # 采样器测试
├── boxes_json/            # 裁剪框缓存
├── checkpoints/          # 模型保存目录
├── weights/             # 预训练权重
├── train.py            # 训练入口
└── test.py            # 评估入口
```

## 🚀 环境配置

### 1. 创建环境
```bash
conda create -n metric python=3.11
conda activate metric
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 核心依赖说明
- **深度学习框架**：torch==2.7.1+cu128, torchvision==0.22.1+cu128
- **度量学习**：pytorch-metric-learning==2.9.0
- **目标检测**：ultralytics==8.3.222
- **数据处理**：opencv-python, numpy, scikit-learn
- **测试框架**：pytest（完整测试套件）

## 📊 数据组织

```
data/
├── train/
│   ├── 举手/          # 视频文件 (.mp4, .avi 等)
│   ├── 玩手机/
│   ├── 正常/
│   └── [其他行为]/
└── val/
    └── [对应类别]/
```

## 🔧 使用指南

### 1. 训练模型
```bash
python train.py --data_root data \
                --output_dir checkpoints/model \
                --p_classes 5 \
                --k_samples 4 \
                --embedding_dim 128 \
                --epochs 30 \
                --margin 0.2
```

**核心参数**：
- `--p_classes`: P-K 采样中的类别数（建议 3-5）
- `--k_samples`: 每类样本数（建议 4-8）
- `--embedding_dim`: 特征维度（128/256/512）
- `--freeze_layers`: 冻结层配置（如 "stem,layer1"）
- `--margin`: Triplet Loss 边界值（0.1-0.5）

### 2. 模型评估
```bash
python test.py --model_path checkpoints/model/best_model.pth \
               --data_root data
```

**输出结果**：
- k-NN 分类准确率和详细报告
- t-SNE 降维可视化图（`val_tsne_plot.png`）
- 特征向量文件（`val_embeddings.pt`）

### 3. 自动裁剪功能

批量裁剪视频：
```bash
python tool/view_cropping.py --input_dir data/train \
                             --output_dir data_cropped/train \
                             --weights weights/yolov11n.pt
```

**裁剪特性**：
- 基于 YOLO 检测人物区域
- 支持裁剪框缓存（`boxes_json/crop_boxes.json`）
- 连续帧检测失败时使用上一帧结果
- 可配置默认裁剪区域和检测阈值

## �️ 技术架构

### 模型设计
- **主干网络**：R(2+1)D_18（Kinetics-400 预训练）
- **特征投影**：全连接层投影到指定维度
- **层级冻结**：支持细粒度层冻结策略

### 训练策略
- **损失函数**：TripletMarginLoss + TripletMarginMiner
- **采样策略**：PKBatchSampler（P类-K样本）
- **优化器**：Adam + 学习率调度
- **数据增强**：随机翻转、亮度/对比度调整

### 评估方法
- **特征提取**：批量提取验证集特征
- **性能评估**：k-NN 分类器（k=5）
- **可视化分析**：t-SNE 降维聚类分析

## � 测试框架

运行完整测试套件：
```bash
pytest -v
```

**测试覆盖**：
- 数据集加载和预处理
- 模型前向传播
- 损失函数计算
- P-K 批次采样器
- YOLO 自动裁剪功能

**测试标记**：
- `pytest -m "not slow"`: 排除耗时测试
- `pytest -m "gpu"`: 仅运行 GPU 测试

## 📈 性能优化

### 数据处理优化
- 预计算裁剪框缓存
- 多进程数据加载
- 智能帧采样策略

### 训练优化
- 梯度累积支持
- 混合精度训练兼容
- 动态学习率调整

## 🤝 开发指南

### 代码规范
- 使用 flake8 进行代码检查（配置：`.flake8`）
- 行长度限制：120 字符
- Git 预提交钩子：`.pre-commit-config.yaml`

### 贡献流程
1. Fork 项目并创建功能分支
2. 添加相应的测试用例
3. 确保所有测试通过：`pytest`
4. 提交 Pull Request

## 📚 扩展功能

- **多模型支持**：可切换不同的视频网络架构
- **在线推理**：支持实时视频流处理
- **数据增强**：可配置的增强策略
- **分布式训练**：多GPU训练支持
