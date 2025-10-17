"""
稳健的视频分类模型训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time
from tqdm import tqdm

# 导入自定义模块
from dataset import VideoDataset
from model import R2Plus1DNet
from loss import ImprovedTripletLoss, ArcFaceLoss, CombinedLoss, CenterLoss
from loss_config import get_loss_config, get_training_config, list_available_configs

# 导入数据预处理类
import random
from PIL import Image

class CustomCropWithNoise:
    def __init__(self, left=1200, top=250, width=1300, height=1500, noise_range=50):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.noise_range = noise_range
    
    def __call__(self, img):
        img_width, img_height = img.size
        
        # 检查图像尺寸是否足够大
        if img_width < self.width or img_height < self.height:
            # 如果图像太小，直接resize到目标尺寸
            return img.resize((self.width, self.height))
        
        # 添加随机噪声抖动
        noise_x = random.randint(-self.noise_range, self.noise_range)
        noise_y = random.randint(-self.noise_range, self.noise_range)
        
        # 计算实际裁剪坐标，确保不超出图像边界
        actual_left = max(0, min(self.left + noise_x, img_width - self.width))
        actual_top = max(0, min(self.top + noise_y, img_height - self.height))
        actual_right = min(img_width, actual_left + self.width)
        actual_bottom = min(img_height, actual_top + self.height)
        
        return img.crop((actual_left, actual_top, actual_right, actual_bottom))

class CustomCropWithoutNoise:
    def __init__(self, left=1200, top=250, width=1300, height=1500):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
    
    def __call__(self, img):
        img_width, img_height = img.size
        
        # 检查图像尺寸是否足够大
        if img_width < self.width or img_height < self.height:
            return img.resize((self.width, self.height))
        
        actual_left = max(0, min(self.left, img_width - self.width))
        actual_top = max(0, min(self.top, img_height - self.height))
        actual_right = min(img_width, actual_left + self.width)
        actual_bottom = min(img_height, actual_top + self.height)
        
        return img.crop((actual_left, actual_top, actual_right, actual_bottom))

def create_data_loaders(data_dir, batch_size=4, num_frames=16, image_size=224, config=None):
    """
    创建数据加载器
    """
    print("创建数据加载器...")
    
    # 从配置获取裁剪参数
    if config is None:
        config = get_main_training_config()
    
    CROP_LEFT = config['CROP_LEFT']
    CROP_TOP = config['CROP_TOP']
    CROP_WIDTH = config['CROP_WIDTH']
    CROP_HEIGHT = config['CROP_HEIGHT']
    NOISE_RANGE = config['NOISE_RANGE']

    # 验证集变换
    val_transform = transforms.Compose([
        CustomCropWithoutNoise(
            left=CROP_LEFT, 
            top=CROP_TOP, 
            width=CROP_WIDTH, 
            height=CROP_HEIGHT
        ),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 训练集变换
    train_transform = transforms.Compose([
        CustomCropWithNoise(
            left=CROP_LEFT, 
            top=CROP_TOP, 
            width=CROP_WIDTH, 
            height=CROP_HEIGHT,
            noise_range=NOISE_RANGE
        ),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=config['HORIZONTAL_FLIP_PROB']),
        transforms.ColorJitter(
            brightness=config['BRIGHTNESS'], 
            contrast=config['CONTRAST'], 
            saturation=config['SATURATION'], 
            hue=config['HUE']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集
    train_dataset = VideoDataset(
        root_dir=os.path.join(data_dir, 'train'),
        num_frames=num_frames,
        transform=train_transform
    )

    val_dataset = VideoDataset(
        root_dir=os.path.join(data_dir, 'val'),
        num_frames=num_frames,
        transform=val_transform
    )

    print(f"数据集信息:")
    print(f"  类别: {train_dataset.classes}")
    print(f"  训练样本数: {len(train_dataset)}")
    print(f"  验证样本数: {len(val_dataset)}")

    # 创建数据加载器 - 使用配置参数
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY'],
        drop_last=config['DROP_LAST_TRAIN']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY'],
        drop_last=config['DROP_LAST_VAL']
    )

    return train_loader, val_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, config):
    """
    训练一个epoch
    """
    model.train()
    running_loss = 0.0
    running_loss_components = {}  # 用于记录组合损失的各个组件
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} - Training')
    
    for batch_idx, (videos, labels) in enumerate(progress_bar):
        try:
            videos = videos.to(device)
            labels = labels.to(device)
            
            # 前向传播
            embeddings = model(videos)
            
            # 处理不同损失函数的返回值
            if isinstance(criterion, (ArcFaceLoss,)):
                loss, logits = criterion(embeddings, labels)
            elif isinstance(criterion, CombinedLoss):
                loss, loss_dict, logits = criterion(embeddings, labels)
            else:  # ImprovedTripletLoss or other single return loss
                loss = criterion(embeddings, labels)
                logits = None
            
            # 检查损失是否为有效数值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到无效损失值 {loss.item()}, 跳过此批次")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['GRADIENT_CLIP_MAX_NORM'])
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # 记录组合损失的各个组件
            if isinstance(criterion, CombinedLoss) and 'loss_dict' in locals():
                for component_name, component_loss in loss_dict.items():
                    if component_name not in running_loss_components:
                        running_loss_components[component_name] = 0.0
                    running_loss_components[component_name] += component_loss.item()
            
            # 更新进度条
            postfix_dict = {
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            }
            
            # 添加组件损失信息到进度条
            if running_loss_components:
                for name, comp_loss in running_loss_components.items():
                    postfix_dict[f'{name.capitalize()}'] = f'{comp_loss/(batch_idx+1):.4f}'
            
            progress_bar.set_postfix(postfix_dict)
            
        except Exception as e:
            print(f"批次 {batch_idx} 训练时出错: {e}")
            continue
    
    return running_loss / max(num_batches, 1)

def validate_epoch(model, val_loader, criterion, device, epoch, total_epochs, config):
    """
    验证一个epoch
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{total_epochs} - Validation')
    
    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(progress_bar):
            try:
                videos = videos.to(device)
                labels = labels.to(device)
                
                # 前向传播
                embeddings = model(videos)
                
                # 处理不同损失函数的返回值
                if isinstance(criterion, (ArcFaceLoss,)):
                    loss, logits = criterion(embeddings, labels)
                elif isinstance(criterion, CombinedLoss):
                    loss, loss_dict, logits = criterion(embeddings, labels)
                else:  # ImprovedTripletLoss or other single return loss
                    loss = criterion(embeddings, labels)
                    logits = None
                
                # 检查损失是否为有效数值
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    running_loss += loss.item()
                
                # 准确率计算 - 根据损失函数类型选择不同的计算方式
                batch_size = embeddings.size(0)
                
                if isinstance(criterion, (ArcFaceLoss, CombinedLoss)) and logits is not None:
                    # 对于有分类logits的损失函数，使用分类准确率
                    _, predicted = torch.max(logits, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    
                elif batch_size > 1:
                    # 对于度量学习损失函数，使用最近邻准确率
                    for i in range(batch_size):
                        query_embedding = embeddings[i:i+1]
                        query_label = labels[i]
                        
                        # 计算与其他样本的距离
                        distances = torch.cdist(query_embedding, embeddings)
                        distances[0, i] = float('inf')  # 排除自己
                        
                        if distances[0].numel() > 0:
                            nearest_idx = torch.argmin(distances)
                            nearest_label = labels[nearest_idx]
                            
                            if nearest_label == query_label:
                                correct_predictions += 1
                        total_samples += 1
                
                # 更新进度条
                accuracy = (correct_predictions / max(total_samples, 1)) * 100
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
                
            except Exception as e:
                print(f"批次 {batch_idx} 验证时出错: {e}")
                continue
    
    avg_loss = running_loss / max(len(val_loader), 1)
    accuracy = (correct_predictions / max(total_samples, 1)) * 100
    
    return avg_loss, accuracy

def main():
    """
    主训练函数
    """
    # ========== 载入和验证配置参数 ==========
    config = get_main_training_config()
    validate_config(config)
    
    print("开始训练视频分类模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取配置
    print("\n" + "="*50)
    loss_config = get_loss_config(config['LOSS_CONFIG_NAME'])
    training_config = get_training_config(config['TRAINING_CONFIG_NAME'])
    print("="*50)
    
    # 从配置获取参数
    DATA_DIR = config['DATA_DIR']
    NUM_CLASSES = config['NUM_CLASSES']
    EMBEDDING_DIM = training_config['embedding_dim']
    BATCH_SIZE = training_config['batch_size']
    NUM_FRAMES = training_config['num_frames']
    IMAGE_SIZE = training_config['image_size']
    LEARNING_RATE = training_config['learning_rate']
    NUM_EPOCHS = training_config['num_epochs']
    
    # 创建数据加载器
    train_loader, val_loader, classes = create_data_loaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        config=config
    )
    
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # 创建模型
    print("创建模型...")
    model = R2Plus1DNet(
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        pretrained=True
    ).to(device)
    
    # 创建损失函数和优化器
    print("创建损失函数...")
    
    # 从配置创建损失函数
    LOSS_TYPE = loss_config['type']
    loss_params = loss_config['params']
    
    if LOSS_TYPE == 'triplet':
        criterion = ImprovedTripletLoss(**loss_params)
        
    elif LOSS_TYPE == 'arcface':
        criterion = ArcFaceLoss(
            in_features=EMBEDDING_DIM,
            out_features=NUM_CLASSES,
            **loss_params
        )
        
    elif LOSS_TYPE == 'combined':
        criterion = CombinedLoss(
            in_features=EMBEDDING_DIM,
            out_features=NUM_CLASSES,
            **loss_params
        )
    
    # 确保损失函数也移动到正确的设备
    criterion = criterion.to(device)
    print(f"损失函数已移动到设备: {device}")
    
    print(f"使用损失函数: {LOSS_TYPE}")
    
    # 打印损失函数详细信息
    if LOSS_TYPE == 'triplet':
        print(f"  - Triplet Loss参数: margin={criterion.margin}, mining={criterion.mining_strategy}, metric={criterion.distance_metric}")
    elif LOSS_TYPE == 'arcface':
        print(f"  - ArcFace参数: s={criterion.s}, m={criterion.m}, easy_margin={criterion.easy_margin}")
    elif LOSS_TYPE == 'combined':
        print(f"  - 组合损失权重: {criterion.loss_weights}")
        print(f"  - ArcFace参数: s={criterion.arcface_loss.s}, m={criterion.arcface_loss.m}")
        print(f"  - Triplet参数: margin={criterion.triplet_loss.margin}, mining={criterion.triplet_loss.mining_strategy}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=config['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['LR_SCHEDULER_STEP_SIZE'], gamma=config['LR_SCHEDULER_GAMMA'])
    
    # 训练记录
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0.0
    
    print(f"开始训练，共 {NUM_EPOCHS} 个epoch...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        try:
            # 训练
            start_time = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS, config)
            train_time = time.time() - start_time
            
            # 验证
            start_time = time.time()
            val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device, epoch, NUM_EPOCHS, config)
            val_time = time.time() - start_time
            
        except KeyboardInterrupt:
            print(f"\n用户中断训练在epoch {epoch+1}")
            print("保存当前进度...")
            # 保存中断时的模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'interrupted': True
            }, 'interrupted_model.pth')
            print("模型已保存为: interrupted_model.pth")
            break
            
        except Exception as e:
            print(f"Epoch {epoch+1} 出现错误: {e}")
            print("继续下一个epoch...")
            continue
        
        # 更新学习率
        scheduler.step()
        
        # 记录结果
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 输出结果
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练损失: {train_loss:.4f} (耗时: {train_time:.1f}s)")
        print(f"  验证损失: {val_loss:.4f} (耗时: {val_time:.1f}s)")
        print(f"  验证准确率: {val_accuracy:.2f}%")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'classes': classes,
                'loss_type': LOSS_TYPE,
                'hyperparameters': {
                    'embedding_dim': EMBEDDING_DIM,
                    'num_classes': NUM_CLASSES,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE
                }
            }, config['BEST_MODEL_PATH'])
            print(f"  ✓ 保存最佳模型 (准确率: {best_accuracy:.2f}%)")
        
        # 每N个epoch保存一次检查点
        if (epoch + 1) % config['SAVE_CHECKPOINT_EVERY'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'classes': classes,
                'loss_type': LOSS_TYPE
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ 保存检查点: checkpoint_epoch_{epoch+1}.pth")
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"最佳验证准确率: {best_accuracy:.2f}%")
    print(f"最佳模型已保存为: {config['BEST_MODEL_PATH']}")
    
    # 保存训练历史
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'classes': classes,
        'loss_type': LOSS_TYPE,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'num_frames': NUM_FRAMES,
            'image_size': IMAGE_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'embedding_dim': EMBEDDING_DIM,
            'loss_type': LOSS_TYPE
        }
    }, config['TRAINING_HISTORY_PATH'])
    print(f"训练历史已保存为: {config['TRAINING_HISTORY_PATH']}")

def get_main_training_config():
    """
    ========================================================================
                            🔧 训练配置中心 🔧
    ========================================================================
    
    这里包含了所有可以调整的训练参数，您只需要修改这个函数中的参数值
    就可以调整整个训练流程。不再需要在代码各处修改参数！
    
    📋 主要配置类别:
    - 基础配置: 数据路径、类别数量
    - 损失函数配置: 选择不同的损失函数组合
    - 数据预处理: 裁剪、增强等参数
    - 训练超参数: 学习率、权重衰减等
    - 模型保存: 文件路径和保存频率
    
    💡 快速配置指南:
    1. 修改 LOSS_CONFIG_NAME 来选择损失函数
    2. 修改 TRAINING_CONFIG_NAME 来选择训练强度
    3. 调整 CROP_* 参数来适应您的图像
    4. 根据需要调整其他参数
    
    ========================================================================
    """
    config = {
        # ========== 基础配置 ==========
        'DATA_DIR': './data_',                    # 数据目录
        'NUM_CLASSES': 5,                        # 类别数量
        
        # ========== 损失函数配置 ==========
        # 🎯 推荐配置组合:
        # - 'behavior_recognition' + 'small_memory'  ⭐ 行为识别推荐 (组合损失)
        # - 'combined_balanced' + 'default'          ⭐ 平衡性能 (组合损失)
        # - 'combined_classification' + 'default'     🎯 强调分类 (组合损失)
        # - 'small_dataset' + 'small_memory'          💾 小数据集 (组合损失)
        # - 'triplet' + 'fast_prototype'              ⚡ 纯三元组训练
        
        # 当前配置: 渐进式平衡损失 (温和调整权重)
        'LOSS_CONFIG_NAME': 'behavior_progressive',
        
        # 其他可选配置:
        # 'LOSS_CONFIG_NAME': 'behavior_recognition_enhanced',  # 增强版（中等平衡）
        # 'LOSS_CONFIG_NAME': 'behavior_balanced',              # 强平衡版（激进调整）
        # 'LOSS_CONFIG_NAME': 'behavior_recognition',           # 原始配置（权重较小）
        # 'LOSS_CONFIG_NAME': 'behavior_metric_learning',       # 强调度量学习
        # 'LOSS_CONFIG_NAME': 'triplet',                        # 纯三元组训练
        
        # ========== 训练参数配置 ==========
        # 可选: 'default', 'large_batch', 'small_memory', 'fast_prototype'
        'TRAINING_CONFIG_NAME': 'default',
        
        # ========== 数据预处理参数 ==========
        'CROP_LEFT': 1200,                       # 裁剪左边界
        'CROP_TOP': 250,                         # 裁剪上边界
        'CROP_WIDTH': 1300,                      # 裁剪宽度
        'CROP_HEIGHT': 1500,                     # 裁剪高度
        'NOISE_RANGE': 50,                       # 训练时的随机噪声范围
        
        # ========== 数据增强参数 ==========
        'HORIZONTAL_FLIP_PROB': 0.5,            # 水平翻转概率
        'BRIGHTNESS': 0.2,                       # 亮度调整范围
        'CONTRAST': 0.2,                         # 对比度调整范围
        'SATURATION': 0.1,                       # 饱和度调整范围
        'HUE': 0.1,                             # 色调调整范围
        
        # ========== 训练超参数 ==========
        'WEIGHT_DECAY': 1e-4,                   # 权重衰减
        'GRADIENT_CLIP_MAX_NORM': 1.0,          # 梯度裁剪最大范数
        'LR_SCHEDULER_STEP_SIZE': 5,            # 学习率调度步长
        'LR_SCHEDULER_GAMMA': 0.5,              # 学习率衰减系数
        
        # ========== 模型保存配置 ==========
        'SAVE_CHECKPOINT_EVERY': 5,             # 每N个epoch保存检查点
        'BEST_MODEL_PATH': 'best_model.pth',    # 最佳模型保存路径
        'TRAINING_HISTORY_PATH': 'training_history.pth',  # 训练历史保存路径
        
        # ========== 数据加载器配置 ==========
        'NUM_WORKERS': 0,                       # Windows兼容设置
        'PIN_MEMORY': False,                    # 内存固定
        'DROP_LAST_TRAIN': True,               # 训练时丢弃最后一个不完整batch
        'DROP_LAST_VAL': False,                # 验证时不丢弃
        
        # ========== 显示和日志配置 ==========
        'VERBOSE': True,                        # 详细输出
        'PROGRESS_BAR': True,                   # 显示进度条
    }
    
    return config

def validate_config(config):
    """
    验证配置的完整性和合理性
    """
    # 检查必需的配置项
    required_keys = [
        'DATA_DIR', 'NUM_CLASSES', 'LOSS_CONFIG_NAME', 'TRAINING_CONFIG_NAME',
        'CROP_LEFT', 'CROP_TOP', 'CROP_WIDTH', 'CROP_HEIGHT'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"配置缺少必需项: {missing_keys}")
    
    # 检查数值的合理性
    if config['NUM_CLASSES'] <= 0:
        raise ValueError("NUM_CLASSES 必须大于0")
    
    if config['CROP_WIDTH'] <= 0 or config['CROP_HEIGHT'] <= 0:
        raise ValueError("裁剪尺寸必须大于0")
    
    if not os.path.exists(config['DATA_DIR']):
        print(f"警告: 数据目录不存在: {config['DATA_DIR']}")
    
    print("✓ 配置验证通过")

if __name__ == "__main__":
    main()