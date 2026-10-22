# coding: utf-8
"""
分支对齐工具函数
用于多分支对齐对比学习的辅助功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_feature_alignment(seg_features, depth_features, save_path=None):
    """
    可视化特征对齐效果
    Args:
        seg_features: Seg分支特征 [B, D]
        depth_features: Depth分支特征 [B, D]
        save_path: 保存路径
    """
    # 计算特征相似度
    similarity = F.cosine_similarity(seg_features, depth_features, dim=1)
    
    # 绘制相似度分布
    plt.figure(figsize=(10, 6))
    plt.hist(similarity.cpu().numpy(), bins=50, alpha=0.7, label='Cosine Similarity')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Feature Alignment Distribution')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
    
    return similarity.mean().item()


def compute_alignment_metrics(seg_features, depth_features):
    """
    计算对齐指标
    Args:
        seg_features: Seg分支特征 [B, D]
        depth_features: Depth分支特征 [B, D]
    Returns:
        metrics: 对齐指标字典
    """
    metrics = {}
    
    # 1. 余弦相似度
    cosine_sim = F.cosine_similarity(seg_features, depth_features, dim=1)
    metrics['cosine_similarity'] = cosine_sim.mean().item()
    
    # 2. L2距离
    l2_distance = F.mse_loss(seg_features, depth_features)
    metrics['l2_distance'] = l2_distance.item()
    
    # 3. 特征方差
    seg_var = torch.var(seg_features, dim=0).mean().item()
    depth_var = torch.var(depth_features, dim=0).mean().item()
    metrics['seg_variance'] = seg_var
    metrics['depth_variance'] = depth_var
    
    # 4. 特征相关性
    seg_flat = seg_features.view(seg_features.size(0), -1)
    depth_flat = depth_features.view(depth_features.size(0), -1)
    correlation = torch.corrcoef(torch.stack([seg_flat.mean(0), depth_flat.mean(0)]))[0, 1]
    metrics['correlation'] = correlation.item() if not torch.isnan(correlation) else 0.0
    
    return metrics


def visualize_tsne_features(seg_features, depth_features, labels=None, save_path=None):
    """
    使用t-SNE可视化特征分布
    Args:
        seg_features: Seg分支特征 [B, D]
        depth_features: Depth分支特征 [B, D]
        labels: 样本标签 [B]
        save_path: 保存路径
    """
    # 合并特征
    all_features = torch.cat([seg_features, depth_features], dim=0)
    
    # 生成标签
    if labels is None:
        labels = torch.arange(seg_features.size(0))
    
    feature_labels = torch.cat([labels, labels], dim=0)
    branch_labels = torch.cat([torch.zeros(seg_features.size(0)), torch.ones(depth_features.size(0))], dim=0)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features.cpu().numpy())
    
    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    # 子图1：按样本分组
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=feature_labels.numpy(), cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Features by Sample')
    
    # 子图2：按分支分组
    plt.subplot(1, 3, 2)
    colors = ['red' if x == 0 else 'blue' for x in branch_labels.numpy()]
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.7)
    plt.title('Features by Branch (Red: Seg, Blue: Depth)')
    
    # 子图3：对齐效果
    plt.subplot(1, 3, 3)
    seg_2d = features_2d[:seg_features.size(0)]
    depth_2d = features_2d[seg_features.size(0):]
    
    plt.scatter(seg_2d[:, 0], seg_2d[:, 1], c='red', alpha=0.7, label='Seg')
    plt.scatter(depth_2d[:, 0], depth_2d[:, 1], c='blue', alpha=0.7, label='Depth')
    
    # 绘制连接线
    for i in range(min(10, seg_features.size(0))):  # 只显示前10个样本的连接线
        plt.plot([seg_2d[i, 0], depth_2d[i, 0]], [seg_2d[i, 1], depth_2d[i, 1]], 'k-', alpha=0.3)
    
    plt.legend()
    plt.title('Feature Alignment Visualization')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def analyze_alignment_progress(alignment_history, save_path=None):
    """
    分析对齐进度
    Args:
        alignment_history: 对齐损失历史记录
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 4))
    
    # 子图1：对齐损失变化
    plt.subplot(1, 3, 1)
    plt.plot(alignment_history)
    plt.xlabel('Epoch')
    plt.ylabel('Alignment Loss')
    plt.title('Alignment Loss Progress')
    plt.grid(True)
    
    # 子图2：损失趋势
    plt.subplot(1, 3, 2)
    if len(alignment_history) > 10:
        # 计算移动平均
        window_size = min(10, len(alignment_history) // 4)
        moving_avg = np.convolve(alignment_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg)
        plt.xlabel('Epoch')
        plt.ylabel('Moving Average Alignment Loss')
        plt.title('Alignment Loss Trend')
        plt.grid(True)
    
    # 子图3：损失分布
    plt.subplot(1, 3, 3)
    plt.hist(alignment_history, bins=30, alpha=0.7)
    plt.xlabel('Alignment Loss')
    plt.ylabel('Frequency')
    plt.title('Alignment Loss Distribution')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compute_branch_consistency(seg_features_list, depth_features_list):
    """
    计算分支一致性
    Args:
        seg_features_list: Seg分支多尺度特征列表
        depth_features_list: Depth分支多尺度特征列表
    Returns:
        consistency_metrics: 一致性指标
    """
    consistency_metrics = {}
    
    for i, (seg_feat, depth_feat) in enumerate(zip(seg_features_list, depth_features_list)):
        # 计算当前尺度的一致性
        alignment_metrics = compute_alignment_metrics(seg_feat, depth_feat)
        
        for key, value in alignment_metrics.items():
            consistency_metrics[f'scale_{i}_{key}'] = value
    
    # 计算跨尺度一致性
    if len(seg_features_list) > 1:
        cross_scale_consistency = []
        for i in range(len(seg_features_list) - 1):
            seg_consistency = F.cosine_similarity(
                seg_features_list[i].mean(0), seg_features_list[i+1].mean(0)
            ).item()
            depth_consistency = F.cosine_similarity(
                depth_features_list[i].mean(0), depth_features_list[i+1].mean(0)
            ).item()
            cross_scale_consistency.append((seg_consistency + depth_consistency) / 2)
        
        consistency_metrics['cross_scale_consistency'] = np.mean(cross_scale_consistency)
    
    return consistency_metrics


def generate_alignment_report(model, dataloader, device, save_path=None):
    """
    生成对齐报告
    Args:
        model: 增强版模型
        dataloader: 数据加载器
        device: 设备
        save_path: 保存路径
    """
    model.eval()
    
    all_seg_features = []
    all_depth_features = []
    alignment_losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            source = batch['source'].to(device)
            mask = batch['mask'].to(device)
            depth = batch['depth'].to(device)
            labels = torch.arange(source.size(0)).to(device)
            
            # 前向传播
            output = model(source, mask, depth, labels)
            
            if len(output) >= 5:
                img, img_seg, img_dep, alpha, alignment_loss = output
                alignment_losses.append(alignment_loss.item())
            
            # 提取特征用于分析
            # 这里需要根据实际模型结构调整
            # 暂时使用输出特征作为示例
    
    # 生成报告
    report = {
        'total_samples': len(dataloader.dataset),
        'avg_alignment_loss': np.mean(alignment_losses) if alignment_losses else 0.0,
        'alignment_loss_std': np.std(alignment_losses) if alignment_losses else 0.0,
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
    
    return report


def visualize_attention_weights(attention_weights, save_path=None):
    """
    可视化注意力权重
    Args:
        attention_weights: 注意力权重 [B, H, W]
        save_path: 保存路径
    """
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.mean(0)  # 平均批次维度
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights.cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Attention Weights Visualization')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

