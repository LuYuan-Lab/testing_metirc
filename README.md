# 考场行为识别与监控 AI 系统

基于深度学习和度量学习的视频理解系统，用于识别考场中的异常行为。采用 R(2+1)D 视频网络提取时空特征，通过 Triplet Loss 和 P-K 采样策略训练判别性特征表示。

## 🎯 核心特性

- **时空特征提取**：R(2+1)D 视频识别模型，有效捕捉时空特征
- **度量学习训练**：Triplet Loss + Semi-hard/Hard 挖掘策略
- **智能批次采样**：P-K 采样确保有效的三元组构建
- **YOLO 自动裁剪**：基于目标检测的视频区域智能裁剪
- **目标跟踪集成**：支持高级目标跟踪，丰富参数控制
- **专业可视化工具**：独立的检测和跟踪可视化系统
- **全面评估体系**：k-NN 分类 + t-SNE 可视化 + 完整测试套件

## 📁 项目结构

```
.
├── model/                    # 模型实现
│   ├── ResNetModel.py       # 自定义 R(2+1)D 视频网络
│   ├── model.py            # torchvision 接口封装
│   └── CustomModel2.py     # 实验性模型
├── tool/                     # 核心工具集
│   ├── dataset.py          # 视频数据集与预处理
│   ├── loss.py            # Triplet Loss 封装
│   ├── video_crop_processor.py  # YOLO 检测与裁剪核心
│   ├── tracker.py         # 目标跟踪模块
│   └── __init__.py       # 模块导入
├── visualization/          # 专业可视化工具
│   ├── object_detection_visualization.py  # 目标检测可视化
│   └── tracking_visualization.py         # 目标跟踪可视化
├── tests/                  # 测试套件
│   ├── test_dataset.py   # 数据集测试
│   ├── test_model.py     # 模型测试
│   ├── test_loss.py      # 损失函数测试
│   └── test_sampler.py   # 采样器测试
├── boxes_json/            # 裁剪框缓存
├── checkpoints/          # 模型保存目录
├── weights/             # 预训练权重
├── videos/              # 示例视频
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

## 🎬 专业可视化工具

### 目标检测可视化

**快速开始**：
```bash
# 基础检测可视化
python visualization/object_detection_visualization.py --input videos/111.mp4

```

**丰富参数控制**：
- **检测参数**：置信度阈值、目标类别、最大检测数、面积过滤
- **可视化参数**：边框厚度、字体大小、颜色方案、显示选项
- **高级选项**：重叠过滤、排序方式、裁剪边距

### 目标跟踪可视化

**快速开始**：
```bash
# 基础跟踪可视化
python visualization/tracking_visualization.py --input videos/111.mp4

# 调整跟踪敏感度
python visualization/tracking_visualization.py --conf 0.5 --track-conf 0.4 --min-hits 2

# 修改关联度量和轨迹显示
python visualization/tracking_visualization.py --association-metric iou --trail-length 50 --fade-trail

# 检测所有类别的长轨迹跟踪
python visualization/tracking_visualization.py --target-class all --trail-length 60 --color-scheme confidence_based
```

**高级跟踪控制**：
- **检测参数**：检测置信度、跟踪置信度、目标类别限制
- **跟踪参数**：关联度量（cosine/iou/euclidean）、卡尔曼滤波、最大年龄
- **可视化参数**：轨迹长度、颜色方案、渐变效果、厚度控制
- **输出控制**：保存轨迹图、详细统计信息

### 参数优化指南

**置信度调试**：
```bash
# 测试不同置信度效果
python visualization/object_detection_visualization.py --conf 0.3 --max-frames 10  # 低阈值
python visualization/object_detection_visualization.py --conf 0.5 --max-frames 10  # 中阈值  
python visualization/object_detection_visualization.py --conf 0.7 --max-frames 10  # 高阈值
```

**输出文件说明**：
- **可视化视频**：带标注的检测/跟踪结果
- **示例图片**：中间帧的可视化效果
- **轨迹图片**：（跟踪模式）完整的运动轨迹图
- **统计文件**：JSON格式的详细参数和结果统计

## 🏗️ 技术架构

### 模型设计
- **主干网络**：R(2+1)D_18（Kinetics-400 预训练）
- **特征投影**：全连接层投影到指定维度
- **层级冻结**：支持细粒度层冻结策略

### 训练策略
- **损失函数**：TripletMarginLoss + TripletMarginMiner
- **采样策略**：PKBatchSampler（P类-K样本）
- **优化器**：Adam + 学习率调度
- **数据增强**：随机翻转、亮度/对比度调整

### 检测与跟踪架构
```
训练阶段:
AutoCropper → VideoCropProcessor → VideoDataset → 模型训练

可视化阶段:
AutoCropper → detect_with_details() → 丰富参数检测
VideoTracker → update() → 参数化跟踪 → 专业可视化
```

### 评估方法
- **特征提取**：批量提取验证集特征
- **性能评估**：k-NN 分类器（k=5）
- **可视化分析**：t-SNE 降维聚类分析

## 🧪 测试框架

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

### 检测与跟踪优化
- 置信度双重过滤机制
- 参数化跟踪配置
- 内存高效的轨迹管理
- 批量处理支持

### 训练优化
- 梯度累积支持
- 混合精度训练兼容
- 动态学习率调整

## 🛠️ 开发与扩展

### 代码规范
- 使用 flake8 进行代码检查（配置：`.flake8`）
- 行长度限制：120 字符
- Git 预提交钩子：`.pre-commit-config.yaml`

### 扩展功能示例

**自定义检测参数**：
```python
from tool.video_crop_processor import AutoCropper

detector = AutoCropper("weights/yolov11n.pt")
detections = detector.detect_with_details(
    frame=frame,
    target_class="person",
    confidence_threshold=0.5,
    max_detections=10,
    min_box_area=500,
    filter_overlapping=True,
    sort_by="confidence"
)
```

**自定义跟踪配置**：
```python
from tool.tracker import TrackingConfig, VideoTracker

config = TrackingConfig(
    enable_tracking=True,
    track_low_thresh=0.3,
    track_high_thresh=0.7,
    track_buffer=50
)
tracker = VideoTracker(config)
```

### 贡献流程
1. Fork 项目并创建功能分支
2. 添加相应的测试用例
3. 确保所有测试通过：`pytest`
4. 提交 Pull Request

## 📚 相关资源

- **论文参考**：R(2+1)D Networks for Video Understanding
- **预训练模型**：Kinetics-400 数据集
- **YOLO模型**：YOLOv11n 轻量级检测模型
- **跟踪算法**：ByteTrack + BoT-SORT

## 🎯 使用建议

### 训练阶段
1. 使用 `train.py` 进行模型训练
2. 通过 `test.py` 评估模型性能
3. 利用自动裁剪功能预处理数据

### 推理阶段
1. 使用 `object_detection_visualization.py` 进行检测分析
2. 使用 `tracking_visualization.py` 进行跟踪分析
3. 根据应用场景调整相应参数

### 调试技巧
- 使用 `--max-frames` 限制处理帧数进行快速测试
- 通过统计文件分析参数效果
- 利用示例图片快速查看可视化效果

---

本系统为考场行为监控提供了完整的深度学习解决方案，从数据预处理到模型训练，再到专业的可视化分析工具，为研究者和开发者提供了强大而灵活的工具集。
