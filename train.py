import argparse
import os
import random
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from model.ResNetModel import R2Dmodel

# 导入之前编写的模块
from tool.dataset import VideoDataset
from tool.loss import TripletLossWrapper

mp.set_start_method("spawn", force=True)


# -----------------------------------------------------------------------------
# 1. PK Batch Sampler - 这是三元组训练的关键部分
# -----------------------------------------------------------------------------
class PKBatchSampler(Sampler):
    """
    P-K Batch Sampler.
    Ensures each batch contains K samples from P different classes.
    """

    def __init__(self, dataset, p, k):
        self.p = p
        self.k = k
        self.dataset_size = len(dataset)

        # 将标签映射到索引
        self.labels_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.video_files):
            self.labels_to_indices[label].append(idx)

        self.labels = list(self.labels_to_indices.keys())

        # 过滤掉样本数少于 K 的类别
        self.labels = [
            label for label in self.labels if len(self.labels_to_indices[label]) >= k
        ]

    def __iter__(self):
        # 计算每个 epoch 的迭代次数
        num_batches = self.dataset_size // (self.p * self.k)

        for _ in range(num_batches):
            # 1. 随机选择 P 个类别
            selected_classes = random.sample(self.labels, self.p)

            batch_indices = []
            for class_label in selected_classes:
                # 2. 对每个类别，随机选择 K 个样本的索引
                indices_for_class = self.labels_to_indices[class_label]
                selected_indices = random.sample(indices_for_class, self.k)
                batch_indices.extend(selected_indices)

            yield batch_indices

    def __len__(self):
        return self.dataset_size // (self.p * self.k)


# -----------------------------------------------------------------------------
# 2. 训练和验证函数
# -----------------------------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch_num):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num} [Train]")

    for videos, labels in progress_bar:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()

        embeddings = model(videos)
        loss = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(train_loader)


def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(val_loader, desc="Validating")

    with torch.no_grad():
        for videos, labels in progress_bar:
            videos, labels = videos.to(device), labels.to(device)

            embeddings = model(videos)
            loss = loss_fn(embeddings, labels)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(val_loader)


# -----------------------------------------------------------------------------
# 3. 主函数
# -----------------------------------------------------------------------------
def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建数据集
    train_dataset = VideoDataset(
        data_root=args.data_root, mode="train", num_frames=args.num_frames
    )
    val_dataset = VideoDataset(
        data_root=args.data_root, mode="val", num_frames=args.num_frames
    )

    # 创建采样器和数据加载器
    print(f"Creating PK Sampler with P={args.p_classes}, K={args.k_samples}")
    train_sampler = PKBatchSampler(train_dataset, p=args.p_classes, k=args.k_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # 验证集不需要PK采样，使用普通DataLoader即可
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.p_classes * args.k_samples,  # 可以使用和训练时相近的batch size
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 创建模型、损失函数和优化器
    model = R2Dmodel(
        embedding_dim=args.embedding_dim,
        pretrained=True,
        freeze_layers=args.freeze_layers,
    ).to(device)
    loss_fn = TripletLossWrapper(margin=args.margin, miner_type="semihard").to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 开始训练
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        avg_val_loss = validate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        scheduler.step()

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}"
            )

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a video embedding model using Triplet Loss."
    )

    parser.add_argument(
        "--data_root", type=str, default="data", help="Path to the root data directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/yolodetect",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Total number of epochs to train."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=30,
        help="Number of frames to sample from each video.",
    )

    # PK Sampler 参数
    parser.add_argument(
        "--p_classes", type=int, default=5, help="P: Number of classes per batch."
    )
    parser.add_argument(
        "--k_samples",
        type=int,
        default=4,
        help="K: Number of samples per class in a batch.",
    )

    # Model & Loss 参数
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Dimension of the output embedding vector.",
    )
    parser.add_argument(
        "--margin", type=float, default=0.2, help="Margin for the triplet loss."
    )
    parser.add_argument(
        "--freeze_layers",
        type=list,
        default=["stem", "layer1", "layer2", "layer3"],
        help="Whether to freeze the layers of the model.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )

    args = parser.parse_args()
    main(args)
