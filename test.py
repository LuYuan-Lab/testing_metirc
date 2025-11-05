import argparse
import os

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.ResNetModel import R2Dmodel

# 导入我们之前编写的模块
from tool.dataset import VideoDataset


@torch.no_grad()
def extract_embeddings(model, data_loader, device):
    """
    遍历 DataLoader，提取所有视频的特征向量和标签。
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for videos, labels in tqdm(data_loader, desc="Extracting embeddings"):
        videos = videos.to(device)

        embeddings = model(videos)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_embeddings, all_labels


def evaluate_knn(train_embeddings, train_labels, val_embeddings, val_labels, class_names):
    """
    使用 k-NN 分类器评估特征向量的质量。
    """
    print("\n--- k-NN Classification Report ---")
    print(f"Fitting k-NN (k=5) on {len(train_embeddings)} train samples...")

    # 将 PyTorch Tensors 转换为 NumPy arrays
    X_train = train_embeddings.numpy()
    y_train = train_labels.numpy()
    X_test = val_embeddings.numpy()
    y_test = val_labels.numpy()

    # 1. 训练 k-NN 分类器
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)

    # 2. 在验证集上进行预测
    print(f"Predicting on {len(val_embeddings)} validation samples...")
    y_pred = knn.predict(X_test)

    # 3. 计算并打印准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nValidation k-NN Accuracy: {accuracy * 100:.2f}%")

    # 4. 打印详细的分类报告
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)


def visualize_tsne(embeddings, labels, class_names, filename="tsne_plot.png"):
    """
    使用 t-SNE 将特征向量降维并可视化。
    """
    print("\n--- t-SNE Visualization ---")
    print("Running t-SNE... (This may take a minute)")

    # 解决matplotlib中文乱码问题
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题
    except Exception as e:
        print(f"Warning: Could not set Chinese font 'SimHei'. Plot labels may be garbled. Error: {e}")
        print("Please ensure you have a Chinese font like 'SimHei' installed.")

    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()

    # 1. 运行 t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
        n_jobs=-1,
    )
    tsne_results = tsne.fit_transform(embeddings_np)

    # 2. 绘制 t-SNE 图像
    plt.figure(figsize=(12, 10))
    num_classes = len(class_names)
    cmap = plt.get_cmap("jet", num_classes)

    for i, class_name in enumerate(class_names):
        class_indices = labels_np == i
        plt.scatter(
            tsne_results[class_indices, 0],
            tsne_results[class_indices, 1],
            c=([cmap(i / (num_classes - 1))] if num_classes > 1 else [cmap(0)]),  # 修正cmap的使用
            label=class_name,
            alpha=0.7,
        )

    plt.title("t-SNE Visualization of Video Embeddings (on Validation Set)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    if num_classes <= 20:  # 类别太多时图例会很乱
        plt.legend(loc="best", markerscale=2.0)
    plt.grid(True)

    # 3. 保存图像
    plt.savefig(filename)
    print(f"t-SNE plot saved to {filename}")


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据集和 DataLoader
    # 我们需要 'train' 集来训练 k-NN，'val' 集来测试
    train_dataset = VideoDataset(data_root=args.data_root, mode="train", num_frames=args.num_frames)
    val_dataset = VideoDataset(data_root=args.data_root, mode="val", num_frames=args.num_frames)

    # 从数据集中获取类别名称
    class_names = list(train_dataset.class_to_idx.keys())
    print(f"Found {len(class_names)} classes: {class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 2. 加载训练好的模型
    print(f"Loading trained model from {args.model_path}...")
    model = R2Dmodel(embedding_dim=args.embedding_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # 3. 提取特征向量
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    val_embeddings, val_labels = extract_embeddings(model, val_loader, device)

    # 4. (可选) 保存特征向量以备后用
    output_path = os.path.join(os.path.dirname(args.model_path), "val_embeddings.pt")
    torch.save({"embeddings": val_embeddings, "labels": val_labels}, output_path)
    print(f"Validation embeddings saved to {output_path}")

    # 5. 运行 k-NN 评估
    evaluate_knn(train_embeddings, train_labels, val_embeddings, val_labels, class_names)

    # 6. 运行 t-SNE 可视化 (在 'val' 集上)
    plot_path = os.path.join(os.path.dirname(args.model_path), "val_tsne_plot.png")
    visualize_tsne(val_embeddings, val_labels, class_names, filename=plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained video embedding model.")

    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/yolodetect/best_model.pth",
        help="Path to the saved best model checkpoint.",
    )
    parser.add_argument("--data_root", type=str, default="data", help="Path to the root data directory.")

    parser.add_argument(
        "--num_frames",
        type=int,
        default=30,
        help="Number of frames to sample from each video.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Dimension of the output embedding vector.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for extraction (can be larger than training).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )

    args = parser.parse_args()
    main(args)
