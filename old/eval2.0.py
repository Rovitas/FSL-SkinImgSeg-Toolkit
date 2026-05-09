# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from scipy.ndimage import binary_erosion
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings("ignore")

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

def average_surface_distance(pred, target):
    """
    计算平均表面距离 (Average Surface Distance)
    ASD = (sum_{p∈∂P} min_{g∈∂G} d(p,g) + sum_{g∈∂G} min_{p∈∂P} d(g,p)) / (|∂P| + |∂G|)
    """
    device = pred.device
    N = pred.shape[0]
    asd_distances = []
    
    for b in range(N):
        pred_binary = (pred[b, 0] > 0.5).cpu().numpy().astype(np.uint8)
        target_binary = (target[b, 0] > 0.5).cpu().numpy().astype(np.uint8)
        
        # 计算边界
        pred_boundary = get_boundary(pred_binary)
        target_boundary = get_boundary(target_binary)
        
        if pred_boundary.sum() == 0 and target_boundary.sum() == 0:
            # 两者都没有边界，距离为0
            asd_distances.append(0.0)
            continue
        elif pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            # 其中一个没有边界，返回最大距离
            H, W = pred_binary.shape
            max_dist = np.sqrt(H*H + W*W)
            asd_distances.append(max_dist)
            continue
        
        # 计算距离变换
        target_dist = distance_transform_edt(1 - target_boundary)
        pred_dist = distance_transform_edt(1 - pred_boundary)
        
        # 计算平均表面距离
        pred_to_target_dist = target_dist[pred_boundary.astype(bool)].mean()
        target_to_pred_dist = pred_dist[target_boundary.astype(bool)].mean()
        
        asd = (pred_to_target_dist + target_to_pred_dist) / 2.0
        asd_distances.append(asd)
    
    return np.mean(asd_distances)

def get_boundary(binary_mask):
    """
    获取二值掩码的边界
    """

    eroded = binary_erosion(binary_mask)
    boundary = binary_mask - eroded
    return boundary

def boundary_f1_score(pred, target, threshold=0.5):
    """
    计算Boundary F1 Score
    """
    device = pred.device
    N = pred.shape[0]
    bf_scores = []
    
    for b in range(N):
        pred_binary = (pred[b, 0] > threshold).cpu().numpy().astype(np.uint8)
        target_binary = (target[b, 0] > threshold).cpu().numpy().astype(np.uint8)
        
        pred_boundary = get_boundary(pred_binary)
        target_boundary = get_boundary(target_binary)
        
        if pred_boundary.sum() == 0 and target_boundary.sum() == 0:
            bf_scores.append(1.0)  # 完美匹配
            continue
        elif pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            bf_scores.append(0.0)
            continue
        
        # 计算交集
        intersection = np.logical_and(pred_boundary, target_boundary).sum()
        
        # 计算F1分数
        precision = intersection / pred_boundary.sum()
        recall = intersection / target_boundary.sum()
        
        if precision + recall == 0:
            bf_score = 0.0
        else:
            bf_score = 2 * precision * recall / (precision + recall)
        bf_scores.append(bf_score)
    
    return np.mean(bf_scores)

def calculate_metrics(pred, target, eps=1e-6):
    """
    计算所有定量指标
    """
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
    probs_flat = probs.flatten()

    # 计算TP, TN, FP, FN
    TP = (pred_flat * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()

    # 计算 Dice 系数
    intersection = TP
    dice = (2. * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)
    # print(f"DSC: {dice.item()}")

    # 计算 IoU
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    # print(f"IoU: {iou.item()}")

    # 计算准确率
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    # 计算召回率 (Sensitivity/Recall)
    recall = TP / (TP + FN + eps)
    # 计算精确率 (Precision)
    precision = TP / (TP + FP + eps)
    # print(f"ACC: {acc}")
    
    # 计算 PR-AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(
            target_flat.cpu().numpy(),
            probs_flat.cpu().numpy()
        )
        pr_auc = auc(recall_curve, precision_curve)
    except:
        pr_auc = float('nan')
    # print(f"PR_AUC: {pr_auc}")

    # # 计算 Hausdorff 距离
    # hd = directed_hausdorff_distance(pred_binary, target_binary)
    # print(f"HD: {hd.item()}")
    
    # 计算 Average Surface Distance
    asd = average_surface_distance(pred_binary, target_binary)
    # print(f"ASD: {asd}")
    
    # 计算 Boundary F1 Score
    bf_score = boundary_f1_score(pred_binary, target_binary)
    print(f"BFscore: {bf_score}")

    return {
        'DSC': dice.item(),
        'IoU': iou.item(),
        'Accuracy': acc.item(),
        'Recall': recall.item(),
        'Precision': precision.item(),
        # 'HD': hd.item(),
        'PR_AUC': pr_auc,
        'ASD': asd,
        'Boundary_F1': bf_score
    }

def evaluate_single_model(model_path, val_loader, device):

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

    return metrics

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
        {"name": "Baseline_seed_38"},
        {"name": "Baseline_seed_39"},
        {"name": "Baseline_seed_40"},
        {"name": "Baseline_seed_48"},
        {"name": "AdamW_wd_1e-05[Seed_48]"},
        {"name": "AdamW_wd_1e-05[Seed_50]"},
        {"name": "AdamW_wd_5e-05[Seed_48]"},
        {"name": "AdamW_wd_5e-05[Seed_49]"},
        {"name": "AdamW_wd_1e-04[Seed_48]"},
        {"name": "AdamW_wd_1e-04[Seed_49]"},
        # 添加更多模型...
    ]
    # ==============================
    
    # === 开始评估 ===
    results = []

    for exp in experiments:
        m_path = os.path.join(r"D:\Work\Python\_MSDT\saved_results", str(exp['name']), "models\\best_model.pth")
        print(f"Path: {m_path}")
        try:
            result = evaluate_single_model(m_path, test_loader, device)
            header = f"{'Experiment Name':<30} | {'Dice':<8} | {'IoU':<8} | {'Acc':<8} | {'Recall':<8} | {'Precision':<8} | {'PR-AUC':<8} | {'ASD':<8} | {'Boundary_F1':<8}"
            print(header)
            print("-" * len(header))
            pr_auc_str = f"{result['PR_AUC']:.4f}" if not np.isnan(result['PR_AUC']) else "N/A"
            print(f"{exp['name']:<30} | {result['DSC']:<8.4f} | {result['IoU']:<8.4f} | {result['PR_AUC']:<8.4f} | {result['Recall']:<8.4f} | {result['Precision']:<8.4f} | {pr_auc_str:<8} | {result['ASD']:<8.4f} | {result['Boundary_F1']:<8.4f}")
            results.append((exp["name"], result['DSC'], result['IoU'], result['PR_AUC'], result['Recall'], result['Precision'], result['PR_AUC'], result['ASD'], result['Boundary_F1']))
        except Exception as e:
            print(f"{exp['name']:<30} | ERROR: {str(e)}")

    # === 保存完整结果到文件 ===
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Comprehensive Evaluation Results\n")
        f.write("="*120 + "\n")
        f.write(f"{'Experiment Name':<30} | {'Val Loss':<10} | {'Dice':<8} | {'IoU':<8} | {'Acc':<8} | {'Recall':<8} | {'Prec':<8} | {'HD':<8} | {'PR-AUC':<8} | {'ASD':<8} | {'BF1':<8}\n")
        f.write("-"*120 + "\n")
        for name, loss, dice, iou, acc, recall, prec, hd, pr_auc, asd, bf_score in results:
            pr_auc_str = f"{pr_auc:.4f}" if not np.isnan(pr_auc) else "N/A"
            asd_str = f"{asd:.4f}" if not np.isnan(asd) else "N/A"
            bf_str = f"{bf_score:.4f}" if not np.isnan(bf_score) else "N/A"
            f.write(f"{name:<30} | {loss:<10.4f} | {dice:<8.4f} | {iou:<8.4f} | {acc:<8.4f} | {recall:<8.4f} | {prec:<8.4f} | {hd:<8.4f} | {pr_auc_str:<8} | {asd_str:<8} | {bf_str:<8}\n")
    print(f"\n✅ Complete results saved to: {output_file}")