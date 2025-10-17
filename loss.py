import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        """
        优化的 ArcFace 损失函数
        Args:
            in_features (int): 输入特征的维度，这里是模型的 embedding_dim (256)。
            out_features (int): 输出类别的数量，这里是 num_classes (5)。
            s (float): 半径，用于缩放 logits。
            m (float): 角度裕度 (angular margin)，ArcFace的核心。
            easy_margin (bool): 是否使用简化的数值稳定性处理。
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        # 创建一个权重矩阵，代表每个类别的"中心"向量
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)  # 改为xavier_normal，通常更稳定

        # 预计算角度相关常数，提高效率
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        # 数值稳定性参数
        self.eps = 1e-7

    def forward(self, embedding, label):
        """
        优化的前向传播
        Args:
            embedding (torch.Tensor): 模型输出的嵌入向量，形状为 (B, embedding_dim)。
                                      注意：输入前已经做过L2归一化。
            label (torch.Tensor): 对应的真实标签，形状为 (B,)。
        """
        # 确保embedding已归一化
        embedding = F.normalize(embedding, p=2, dim=1, eps=self.eps)
        
        # 权重L2归一化
        normalized_weight = F.normalize(self.weight, p=2, dim=1, eps=self.eps)
        
        # 计算余弦相似度
        cosine = F.linear(embedding, normalized_weight)
        cosine = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)  # 数值稳定性
        
        # 计算正弦值
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), self.eps, 1.0))
        
        # 计算 cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 数值稳定性处理
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 使用scatter_替代one_hot编码，提高效率
        output = cosine.clone()
        batch_size = embedding.size(0)
        
        # 确保索引张量在正确的设备上
        batch_indices = torch.arange(batch_size, device=embedding.device, dtype=torch.long)
        
        # 只对正确类别应用角度裕度
        output[batch_indices, label] = phi[batch_indices, label]
        
        # 乘以缩放因子 s
        output *= self.s
        
        # 使用标准的交叉熵损失函数计算最终的loss
        loss = F.cross_entropy(output, label, label_smoothing=0.1)  # 添加标签平滑
        return loss, output


class ImprovedTripletLoss(nn.Module):
    def __init__(self, margin=0.3, mining_strategy='batch_hard', distance_metric='euclidean'):
        """
        改进的 Triplet Loss 用于度量学习
        Args:
            margin (float): 边界值，正样本对和负样本对之间的最小距离差
            mining_strategy (str): 挖掘策略，'batch_hard', 'batch_all', 'random'
            distance_metric (str): 距离度量，'euclidean' 或 'cosine'
        """
        super(ImprovedTripletLoss, self).__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.distance_metric = distance_metric
        self.eps = 1e-8
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (torch.Tensor): 嵌入向量，形状为 (B, embedding_dim)
            labels (torch.Tensor): 标签，形状为 (B,)
        """
        # 归一化嵌入向量
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=self.eps)
        
        if self.mining_strategy == 'batch_hard':
            return self._batch_hard_triplet_loss(embeddings, labels)
        elif self.mining_strategy == 'batch_all':
            return self._batch_all_triplet_loss(embeddings, labels)
        else:
            return self._random_triplet_loss(embeddings, labels)
    
    def _batch_hard_triplet_loss(self, embeddings, labels):
        """
        优化的 Batch Hard Triplet Loss
        """
        # 计算距离矩阵
        if self.distance_metric == 'euclidean':
            pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
        else:  # cosine distance
            cosine_sim = torch.mm(embeddings, embeddings.t())
            pairwise_distances = 1 - cosine_sim
        
        # 创建mask
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # 获取每个样本的最难正样本和负样本
        batch_size = embeddings.size(0)
        
        # 对角线设为0（自己和自己的距离）
        pairwise_distances = pairwise_distances + torch.eye(batch_size, device=embeddings.device) * 1e6
        
        # 最难正样本：同类中距离最远的
        mask_anchor_positive = labels_equal.float()
        mask_anchor_positive = mask_anchor_positive - torch.eye(batch_size, device=embeddings.device)
        
        hardest_positive_dist = (pairwise_distances * mask_anchor_positive).max(dim=1)[0]
        
        # 最难负样本：不同类中距离最近的
        mask_anchor_negative = labels_not_equal.float()
        hardest_negative_dist = (pairwise_distances + (1 - mask_anchor_negative) * 1e6).min(dim=1)[0]
        
        # 计算triplet loss，只对有效样本计算
        valid_triplets = (mask_anchor_positive.sum(dim=1) > 0) & (mask_anchor_negative.sum(dim=1) > 0)
        
        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        triplet_loss = triplet_loss[valid_triplets]
        
        return triplet_loss.mean()
    
    def _batch_all_triplet_loss(self, embeddings, labels):
        """
        优化的 Batch All Triplet Loss - 使用向量化操作
        """
        batch_size = embeddings.size(0)
        
        # 计算距离矩阵
        if self.distance_metric == 'euclidean':
            pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
        else:
            cosine_sim = torch.mm(embeddings, embeddings.t())
            pairwise_distances = 1 - cosine_sim
        
        # 创建三元组mask
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # anchor-positive pairs (排除自己)
        mask_anchor_positive = labels_equal.float()
        mask_anchor_positive = mask_anchor_positive - torch.eye(batch_size, device=embeddings.device)
        
        # anchor-negative pairs
        mask_anchor_negative = labels_not_equal.float()
        
        # 获取所有有效的三元组
        anchor_positive_dist = pairwise_distances.unsqueeze(2)  # [batch, batch, 1]
        anchor_negative_dist = pairwise_distances.unsqueeze(1)  # [batch, 1, batch]
        
        # 创建三元组mask
        triplet_mask = mask_anchor_positive.unsqueeze(2) * mask_anchor_negative.unsqueeze(1)
        
        # 计算triplet loss
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        triplet_loss = F.relu(triplet_loss)
        
        # 只保留有效的三元组
        triplet_loss = triplet_loss * triplet_mask
        
        # 计算平均loss
        num_positive_triplets = (triplet_loss > 1e-16).sum()
        
        if num_positive_triplets == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        
        return triplet_loss.sum() / num_positive_triplets.float()
    
    def _random_triplet_loss(self, embeddings, labels):
        """
        优化的随机三元组选择
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        
        # 预先计算所有距离
        if self.distance_metric == 'euclidean':
            pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
        else:
            cosine_sim = torch.mm(embeddings, embeddings.t())
            pairwise_distances = 1 - cosine_sim
        
        total_loss = []
        
        for i in range(batch_size):
            # 找到正样本和负样本
            positive_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
            negative_mask = labels != labels[i]
            
            if not positive_mask.any() or not negative_mask.any():
                continue
            
            # 随机选择正负样本
            positive_indices = torch.where(positive_mask)[0]
            negative_indices = torch.where(negative_mask)[0]
            
            pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
            neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
            
            # 计算triplet loss
            pos_dist = pairwise_distances[i, pos_idx]
            neg_dist = pairwise_distances[i, neg_idx]
            
            loss = F.relu(pos_dist - neg_dist + self.margin)
            total_loss.append(loss)
        
        if len(total_loss) == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        
        return torch.stack(total_loss).mean()


class CenterLoss(nn.Module):
    """
    Center Loss: 用于增强类内聚性
    """
    def __init__(self, num_classes, feature_dim, lambda_c=0.001):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c
        
        # 学习每个类的中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_normal_(self.centers)
        
    def forward(self, features, labels):
        """
        Args:
            features: 特征向量 [batch_size, feature_dim]
            labels: 标签 [batch_size]
        """
        batch_size = features.size(0)
        
        # 计算每个样本到其对应类中心的距离
        centers_batch = self.centers[labels]  # [batch_size, feature_dim]
        
        # 计算中心损失
        center_loss = torch.sum(torch.pow(features - centers_batch, 2), dim=1)
        center_loss = center_loss.mean()
        
        return self.lambda_c * center_loss


class CombinedLoss(nn.Module):
    """
    组合损失函数：结合多种损失以获得更好的性能
    """
    def __init__(self, in_features, out_features, 
                 arcface_s=30.0, arcface_m=0.50,
                 triplet_margin=0.3, triplet_mining='batch_hard',
                 center_lambda=0.001,
                 loss_weights={'arcface': 1.0, 'triplet': 0.1, 'center': 0.01}):
        super(CombinedLoss, self).__init__()
        
        self.arcface_loss = ArcFaceLoss(in_features, out_features, arcface_s, arcface_m)
        self.triplet_loss = ImprovedTripletLoss(triplet_margin, triplet_mining)
        self.center_loss = CenterLoss(out_features, in_features, center_lambda)
        
        self.loss_weights = loss_weights
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: 嵌入向量 [batch_size, feature_dim]
            labels: 标签 [batch_size]
        """
        losses = {}
        
        # ArcFace Loss
        if self.loss_weights.get('arcface', 0) > 0:
            arcface_loss, logits = self.arcface_loss(embeddings, labels)
            losses['arcface'] = self.loss_weights['arcface'] * arcface_loss
        else:
            logits = None
        
        # Triplet Loss
        if self.loss_weights.get('triplet', 0) > 0:
            triplet_loss = self.triplet_loss(embeddings, labels)
            losses['triplet'] = self.loss_weights['triplet'] * triplet_loss
        
        # Center Loss
        if self.loss_weights.get('center', 0) > 0:
            center_loss = self.center_loss(embeddings, labels)
            losses['center'] = self.loss_weights['center'] * center_loss
        
        # 总损失
        total_loss = sum(losses.values())
        
        return total_loss, losses, logits