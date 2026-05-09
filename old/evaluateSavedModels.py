# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from myDataset import MYDataset
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def directed_hausdorff_distance(pred, target):
    """
    计算有向Hausdorff距离（内存高效版本）
    """
    device = pred.device
    N, C, H, W = pred.shape
    
    # 获取前景点坐标
    hd_distances = []
    
    for b in range(N):
        # 获取预测和目标的前景点
        pred_fg = (pred[b] > 0.5).nonzero(as_tuple=False)  # (num_points, 3)
        target_fg = (target[b] > 0.5).nonzero(as_tuple=False)  # (num_points, 3)
        
        if len(pred_fg) == 0 or len(target_fg) == 0:
            # 如果任一图像没有前景点，返回最大可能距离
            max_dist = torch.sqrt(torch.tensor(H*H + W*W, dtype=torch.float32, device=device))
            hd_distances.append(max_dist)
            continue
            
        # 提取x, y坐标（忽略通道维度）
        pred_coords = pred_fg[:, 1:].float()  # (num_pred, 2)
        target_coords = target_fg[:, 1:].float()  # (num_target, 2)
        
        # 计算所有点对之间的距离
        # 使用广播机制避免大内存分配
        pred_expanded = pred_coords.unsqueeze(1)  # (num_pred, 1, 2)
        target_expanded = target_coords.unsqueeze(0)  # (1, num_target, 2)
        
        distances = torch.norm(pred_expanded - target_expanded, dim=2)  # (num_pred, num_target)
        
        # 计算从pred到target的最小距离
        min_dists_pred_to_target = distances.min(dim=1)[0]  # (num_pred,)
        # 计算从target到pred的最小距离
        min_dists_target_to_pred = distances.min(dim=0)[0]  # (num_target,)
        
        # Hausdorff距离是两个方向的最大值
        hd1 = min_dists_pred_to_target.max()
        hd2 = min_dists_target_to_pred.max()
        hd = torch.max(hd1, hd2)
        
        hd_distances.append(hd)
    
    return torch.stack(hd_distances).mean()

def calculate_metrics(pred, target, eps=1e-6):
    # 如果 pred 是 logits，先 sigmoid
    if pred.min() < 0 or pred.max() > 1:
        probs = torch.sigmoid(pred)
    else:
        probs = pred
    
    # 二值化预测结果
    pred_binary = (probs >= 0.5).float()
    target_binary = (target >= 0.5).float()

    # 展平为1D数组用于计算
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()

    # 计算TP, TN, FP, FN
    TP = (pred_flat * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()

    # 计算 Dice 系数
    intersection = TP
    dice = (2. * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)

    # 计算 IoU
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + eps) / (union + eps)

    # 计算准确率
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    # 计算召回率 (Sensitivity/Recall)
    recall = TP / (TP + FN + eps)
    # 计算精确率 (Precision)
    precision = TP / (TP + FP + eps)

    # 计算 AUC（需要原始概率值）
    try:
        auc = roc_auc_score(
            target_flat.cpu().numpy(),
            probs.flatten().cpu().numpy()
        )
    except ValueError:
        # 如果所有标签相同（全0或全1），AUC 无定义
        auc = float('nan')
    
    # 计算 Hausdorff 距离
    hd = directed_hausdorff_distance(pred_binary, target_binary)

    return {
        'DSC': dice.item(),
        'IoU': iou.item(),
        'Accuracy': acc.item(),
        'Recall': recall.item(),
        'Precision': precision.item(),
        'HD': hd.item(),
        'auc': auc
    }

def evaluate_single_model(model_path, val_loader, device):
    from myModel import UNet
    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    # 收集所有预测和标签用于全局指标计算
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # Loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # 保存原始 logits 和 labels
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # 合并所有 batch
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 计算综合指标
    metrics = calculate_metrics(all_preds, all_labels)
    avg_loss = total_loss / total_samples

    return (
        avg_loss,
        metrics['DSC'],
        metrics['IoU'],
        metrics['Accuracy'],
        metrics['Recall'],
        metrics['Precision'],
        metrics['HD'],
        metrics['auc']
    )

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    test_dataset = MYDataset(base_dir=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    print(f"Test set size: {len(test_dataset)}\n")
    # ==========================================

    # === 配置：要评估的模型列表 ===
    experiments = [
        {"name": "Baseline_seed_38", "path": r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_38\models\best_model.pth"},
        {"name": "Baseline_seed_39", "path": r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_39\models\best_model.pth"},
        {"name": "Baseline_seed_40", "path": r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_40\models\best_model.pth"},
        {"name": "Baseline_seed_48", "path": r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_48\models\best_model.pth"},
        {"name": "AdamW_wd_1e-04[Seed_48]", "path": r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-04[Seed_48]\models\best_model.pth"},
        # 添加更多模型...
    ]
    # ==============================

    # === 开始评估 ===
    results = []
    print(f"{'Experiment Name':<30} | {'Val Loss':<10} | {'Dice':<10} | {'IoU':<10} | {'Acc':<10} | {'AUC':<10}")
    print("-" * 100)
    
    for exp in experiments:
        try:
            loss, dice, iou, acc, auc = evaluate_single_model(exp["path"], test_loader, device)
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
            print(f"{exp['name']:<30} | {loss:<10.4f} | {dice:<10.4f} | {iou:<10.4f} | {acc:<10.4f} | {auc_str:<10}")
            results.append((exp["name"], loss, dice, iou, acc, auc))
        except Exception as e:
            print(f"{exp['name']:<30} | ERROR: {str(e)}")

    # === 保存结果到文件 ===
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_file = f"evaluation_results_{timestamp}.txt"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.write("Evaluation Results\n")
    #     f.write("="*100 + "\n")
    #     f.write(f"{'Experiment Name':<30} | {'Val Loss':<10} | {'Dice':<10} | {'IoU':<10} | {'Acc':<10} | {'AUC':<10}\n")
    #     f.write("-"*100 + "\n")
    #     for name, loss, dice, iou, acc, auc in results:
    #         auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    #         f.write(f"{name:<30} | {loss:<10.4f} | {dice:<10.4f} | {iou:<10.4f} | {acc:<10.4f} | {auc_str:<10}\n")
    # print(f"\n✅ Results saved to: {output_file}")
