import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.utils import subgraph, index_to_mask, k_hop_subgraph, mask_to_index

def generate_homo(data):
    labels = data.y
    edge_index = data.edge_index

    neighbors_labels = []
    index_test = mask_to_index(data.test_mask)
    for i in index_test:
        # 获取节点 i 的邻居索引
        neighbors = edge_index[1][edge_index[0] == i]
        # 将邻居标签存入列表
        neighbors_labels.append(labels[neighbors])

    homophilous_mask = torch.zeros(len(index_test), dtype=torch.bool)
    heterophilous_mask = torch.zeros(len(index_test), dtype=torch.bool)

    for i in range(len(index_test)):
        # 获取中心节点 i 的标签和邻居标签
        center_label = labels[index_test[i]]
        neighbor_labels = neighbors_labels[i]

        # 计算同标签邻居的数量
        num_same_label = (neighbor_labels == center_label).sum().item()
        num_neighbors = neighbor_labels.size(0)

        # 判断是同配还是异配
        if num_same_label > num_neighbors / 2:
            homophilous_mask[i] = True
        else:
            heterophilous_mask[i] = True
    return homophilous_mask, heterophilous_mask



def visualization(data, labels):
    label_colors = {0: 'blue', 1: 'red'}

    # 创建散点图
    plt.figure(figsize=(8, 6))
    for point, label in zip(data, labels):
        plt.scatter(point[0], point[1], color=label_colors[label])

    # 添加图例和显示
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('scatter_plot.png', format='png')  # 你可以选择不同的格式，如 'pdf', 'svg' 等
    plt.close()  # 关闭图表以释放内存

def t_sne(data, labels):
    data = np.array(data)

    # 使用 t-SNE 将高维数据降到 2D
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    # 标签对应的颜色
    label_colors = {0: 'blue', 1: 'red'}

    # 创建散点图
    plt.figure(figsize=(8, 6))
    for point, label in zip(data_2d, labels):
        plt.scatter(point[0], point[1], color=label_colors[label], alpha=0.6)

    # 添加轴标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')

    # 保存为文件
    plt.savefig('tsne.png', format='png')
    plt.close()  # 关闭图表以释放内存

def causal_loss(causal_output, target_labels):
    """
    Cross-entropy loss for the causal output compared with one-hot target labels.
    """
    return F.cross_entropy(causal_output, target_labels)

def non_causal_loss(non_causal_output, num_classes=2):
    """
    KL-divergence loss to compare non-causal output with a uniform distribution.
    """
    uniform_dist = torch.full_like(non_causal_output, 1.0 / num_classes)  # uniform distribution [0.5, 0.5] for 2 classes
    non_causal_output_log = F.log_softmax(non_causal_output, dim=0)  # Log softmax for non-causal output
    return F.kl_div(non_causal_output_log, uniform_dist, reduction='batchmean')

def orthogonal_loss(causal_embeddings, non_causal_embeddings):
    """
    Cosine similarity loss to enforce orthogonality between causal and non-causal embeddings.
    """
    causal_norm = F.normalize(causal_embeddings, p=2, dim=0)
    non_causal_norm = F.normalize(non_causal_embeddings, p=2, dim=0)
    cosine_similarity = torch.sum(causal_norm * non_causal_norm)  # Cosine similarity
    return cosine_similarity  # Minimize cosine similarity to enforce orthogonality


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        :param alpha: 类别的权重因子，适用于不平衡数据集
        :param gamma: 调节因子，控制难易样本的惩罚力度
        :param reduction: 损失的聚合方式，'mean' 返回平均损失，'sum' 返回总损失，'none' 不进行聚合
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: 模型的输出 [batch_size, num_classes]
        # targets: 真实的类别标签 [batch_size]

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 计算普通交叉熵损失
        pt = torch.exp(-ce_loss)  # 计算 pt = exp(-CE)，代表预测的正确性概率

        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def remove_empty_lines(text):
    pattern = r"\n\s*\n"  
    return re.sub(pattern, "\n", text)