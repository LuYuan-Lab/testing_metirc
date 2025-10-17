"""
损失函数配置文件
用于快速切换不同的损失函数设置
"""

# 损失函数配置
LOSS_CONFIGS = {
    # 纯Triplet损失 - 适合度量学习任务
    'triplet': {
        'type': 'triplet',
        'params': {
            'margin': 0.3,
            'mining_strategy': 'batch_hard',  # 'batch_hard', 'batch_all', 'random'
            'distance_metric': 'euclidean'    # 'euclidean', 'cosine'
        },
        'description': '纯Triplet损失，适合度量学习和检索任务'
    },
    
    # 纯ArcFace损失 - 适合分类任务
    'arcface': {
        'type': 'arcface',
        'params': {
            's': 30.0,           # 缩放因子
            'm': 0.50,           # 角度裕度
            'easy_margin': False  # 数值稳定性处理方式
        },
        'description': 'ArcFace损失，适合人脸识别等分类任务'
    },
    
    # 组合损失 - 平衡版本
    'combined_balanced': {
        'type': 'combined',
        'params': {
            'arcface_s': 30.0,
            'arcface_m': 0.50,
            'triplet_margin': 0.3,
            'triplet_mining': 'batch_hard',
            'center_lambda': 0.001,
            'loss_weights': {
                'arcface': 1.0,    # 主要分类损失
                'triplet': 0.1,    # 辅助度量损失
                'center': 0.01     # 聚类正则化
            }
        },
        'description': '平衡的组合损失，适合大多数情况'
    },
    
    # 组合损失 - 强调度量学习
    'combined_metric': {
        'type': 'combined',
        'params': {
            'arcface_s': 30.0,
            'arcface_m': 0.50,
            'triplet_margin': 0.3,
            'triplet_mining': 'batch_hard',
            'center_lambda': 0.005,
            'loss_weights': {
                'arcface': 0.7,    # 降低分类损失权重
                'triplet': 0.3,    # 增加度量损失权重
                'center': 0.05     # 增加聚类正则化
            }
        },
        'description': '强调度量学习的组合损失，适合检索和相似度任务'
    },
    
    # 组合损失 - 强调分类
    'combined_classification': {
        'type': 'combined',
        'params': {
            'arcface_s': 64.0,     # 增大缩放因子
            'arcface_m': 0.5,
            'triplet_margin': 0.2,
            'triplet_mining': 'batch_hard',
            'center_lambda': 0.0005,
            'loss_weights': {
                'arcface': 1.5,    # 增加分类损失权重
                'triplet': 0.05,   # 降低度量损失权重
                'center': 0.005    # 降低聚类正则化
            }
        },
        'description': '强调分类的组合损失，适合纯分类任务'
    },
    
    # 小数据集配置
    'small_dataset': {
        'type': 'combined',
        'params': {
            'arcface_s': 20.0,     # 降低缩放因子，避免过拟合
            'arcface_m': 0.3,      # 降低角度裕度
            'triplet_margin': 0.5,  # 增大margin，增强区分度
            'triplet_mining': 'batch_all',  # 使用所有样本对
            'center_lambda': 0.01,
            'loss_weights': {
                'arcface': 0.8,
                'triplet': 0.2,    # 增加triplet权重
                'center': 0.1      # 增加center权重
            }
        },
        'description': '小数据集优化配置，增强泛化能力'
    },
    
    # 行为识别专用配置
    'behavior_recognition': {
        'type': 'combined',
        'params': {
            'arcface_s': 25.0,     # 适中的缩放因子
            'arcface_m': 0.4,      # 适中的角度裕度
            'triplet_margin': 0.3,  
            'triplet_mining': 'batch_hard',
            'center_lambda': 0.005,
            'loss_weights': {
                'arcface': 1.0,    # 主要分类损失
                'triplet': 0.15,   # 适度的度量学习
                'center': 0.02     # 轻微的聚类正则化
            }
        },
        'description': '专为行为识别任务优化的配置'
    },
    
    # 增强版行为识别配置 - 更平衡的权重
    'behavior_recognition_enhanced': {
        'type': 'combined',
        'params': {
            'arcface_s': 25.0,
            'arcface_m': 0.4,
            'triplet_margin': 0.3,
            'triplet_mining': 'batch_hard',
            'center_lambda': 0.1,      # 增大center的内部lambda
            'loss_weights': {
                'arcface': 1.0,        # 保持分类损失
                'triplet': 0.5,        # 显著增大triplet权重
                'center': 0.2          # 增大center权重
            }
        },
        'description': '增强版行为识别配置，triplet和center损失更明显'
    },
    
    # 强调度量学习的配置
    'behavior_metric_learning': {
        'type': 'combined',
        'params': {
            'arcface_s': 20.0,         # 稍微降低arcface
            'arcface_m': 0.3,
            'triplet_margin': 0.4,     # 增大margin
            'triplet_mining': 'batch_hard',
            'center_lambda': 0.05,     # 中等center lambda
            'loss_weights': {
                'arcface': 0.7,        # 降低分类损失权重
                'triplet': 1.0,        # triplet作为主要损失
                'center': 0.3          # 较大的center权重
            }
        },
        'description': '强调度量学习的配置，适合相似度任务'
    },
    
    # 平衡损失配置 - 基于数值范围调整权重
    'behavior_balanced': {
        'type': 'combined',
        'params': {
            'arcface_s': 25.0,
            'arcface_m': 0.4,
            'triplet_margin': 0.3,
            'triplet_mining': 'batch_hard',
            'center_lambda': 0.1,      
            'loss_weights': {
                'arcface': 0.1,        # 大幅降低ArcFace权重
                'triplet': 10.0,       # 大幅提高Triplet权重
                'center': 1.0          # 适中的Center权重
            }
        },
        'description': '数值平衡的损失配置，三种损失贡献相当'
    },
    
    # 渐进平衡配置 - 更温和的调整
    'behavior_progressive': {
        'type': 'combined',
        'params': {
            'arcface_s': 25.0,
            'arcface_m': 0.4,
            'triplet_margin': 0.3,
            'triplet_mining': 'batch_hard',
            'center_lambda': 0.1,      
            'loss_weights': {
                'arcface': 0.3,        # 适度降低ArcFace
                'triplet': 3.0,        # 适度提高Triplet
                'center': 0.4          # 适度提高Center
            }
        },
        'description': '渐进式平衡配置，温和调整权重比例'
    }
}

# 训练超参数配置
TRAINING_CONFIGS = {
    'default': {
        'batch_size': 3,
        'num_frames': 12,
        'image_size': 224,
        'learning_rate': 0.0005,
        'num_epochs': 15,
        'embedding_dim': 128
    },
    
    'large_batch': {
        'batch_size': 8,        # 增大batch size
        'num_frames': 16,       # 增加帧数
        'image_size': 224,
        'learning_rate': 0.001, # 增大学习率
        'num_epochs': 20,
        'embedding_dim': 256    # 增大embedding维度
    },
    
    'small_memory': {
        'batch_size': 2,        # 减小batch size
        'num_frames': 8,        # 减少帧数
        'image_size': 192,      # 减小图像尺寸
        'learning_rate': 0.0003,
        'num_epochs': 25,       # 增加epoch数补偿
        'embedding_dim': 128
    },
    
    'fast_prototype': {
        'batch_size': 4,
        'num_frames': 6,        # 大幅减少帧数
        'image_size': 160,      # 减小图像尺寸
        'learning_rate': 0.001,
        'num_epochs': 10,       # 快速原型验证
        'embedding_dim': 64     # 减小embedding维度
    }
}

def get_loss_config(config_name='combined_balanced'):
    """
    获取损失函数配置
    
    Args:
        config_name (str): 配置名称
        
    Returns:
        dict: 损失函数配置
    """
    if config_name not in LOSS_CONFIGS:
        print(f"警告: 配置 '{config_name}' 不存在，使用默认配置 'combined_balanced'")
        config_name = 'combined_balanced'
    
    config = LOSS_CONFIGS[config_name].copy()
    print(f"使用损失函数配置: {config_name}")
    print(f"描述: {config['description']}")
    
    return config

def get_training_config(config_name='default'):
    """
    获取训练配置
    
    Args:
        config_name (str): 配置名称
        
    Returns:
        dict: 训练配置
    """
    if config_name not in TRAINING_CONFIGS:
        print(f"警告: 配置 '{config_name}' 不存在，使用默认配置 'default'")
        config_name = 'default'
    
    config = TRAINING_CONFIGS[config_name].copy()
    print(f"使用训练配置: {config_name}")
    
    return config

def list_available_configs():
    """
    列出所有可用的配置
    """
    print("可用的损失函数配置:")
    for name, config in LOSS_CONFIGS.items():
        print(f"  - {name}: {config['description']}")
    
    print("\n可用的训练配置:")
    for name, config in TRAINING_CONFIGS.items():
        print(f"  - {name}: batch_size={config['batch_size']}, lr={config['learning_rate']}")

if __name__ == "__main__":
    list_available_configs()