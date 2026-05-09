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

    # 计算 IoU
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + eps) / (union + eps)

    # 计算准确率
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    # 计算召回率 (Sensitivity/Recall)
    recall = TP / (TP + FN + eps)
    # 计算精确率 (Precision)
    precision = TP / (TP + FP + eps)

    # 计算 PR-AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(
            target_flat.cpu().numpy(),
            probs_flat.cpu().numpy()
        )
        pr_auc = auc(recall_curve, precision_curve)
    except:
        pr_auc = float('nan')

    # 计算 Average Surface Distance
    asd = average_surface_distance(pred_binary, target_binary)
    
    # 计算 Boundary F1 Score
    bf_score = boundary_f1_score(pred_binary, target_binary)

    return {
        'DSC': dice.item(),
        'IoU': iou.item(),
        'Accuracy': acc.item(),
        'Recall': recall.item(),
        'Precision': precision.item(),
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
    avg_loss = total_loss / total_samples
    metrics['Val_Loss'] = avg_loss

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
        # {"name": "Baseline_seed_38"},
        # {"name": "Baseline_seed_39"},
        # {"name": "Baseline_seed_40"},
        # {"name": "Baseline_seed_48"},

        # {"name": "Adam_wd_1e-05[Seed_48]"},
        # {"name": "Adam_wd_1e-05[Seed_50]"},

        # {"name": "Adam_wd_5e-05[Seed_48]"},
        # {"name": "Adam_wd_5e-05[Seed_49]"},
        
        # {"name": "Adam_wd_1e-04[Seed_48]"},
        # {"name": "Adam_wd_1e-04[Seed_49]"},

        # {"name": "AdamW_wd_0[Seed_48]"},
        # {"name": "AdamW_wd_0[Seed_49]"},
        # {"name": "AdamW_wd_0[Seed_50]"},

        # {"name": "AdamW_wd_1e-05[Seed_48]"},
        # {"name": "AdamW_wd_1e-05[Seed_49]"},
        # {"name": "AdamW_wd_1e-05[Seed_50]"},

        # {"name": "AdamW_wd_5e-05[Seed_48]"},
        # {"name": "AdamW_wd_5e-05[Seed_49]"},
        # {"name": "AdamW_wd_5e-05[Seed_50]"},

        # {"name": "AdamW_wd_1e-04[Seed_48]"},
        # {"name": "AdamW_wd_1e-04[Seed_49]"},
        # {"name": "AdamW_wd_1e-04[Seed_50]"},

        # {"name": "DiceLoss[Seed_48]"},
        # {"name": "DiceLoss[Seed_49]"},
        # {"name": "DiceLoss[Seed_50]"},

        # {"name": "FocalLoss(a0.25_g2)[Seed_48]"},
        # {"name": "FocalLoss(a0.25_g2)[Seed_49]"},
        # {"name": "FocalLoss(a0.25_g2)[Seed_50]"},

        # {"name": "TverskyLoss(a0.3_b0.7)[Seed_48]"},
        # {"name": "TverskyLoss(a0.3_b0.7)[Seed_49]"},
        # {"name": "TverskyLoss(a0.3_b0.7)[Seed_50]"},

        # {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]"},
        {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_49]"},
        {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_50]"},

        {"name": "TverskyHD(a0.3_b0.7w0.5_0.5)[Seed_48]"},
        {"name": "TverskyHD(a0.3_b0.7w0.5_0.5)[Seed_49]"},
        {"name": "TverskyHD(a0.3_b0.7w0.5_0.5)[Seed_50]"},

        {"name": "TverskyHD(a0.3_b0.7w0.9_0.1)[Seed_48]"},
        {"name": "TverskyHD(a0.3_b0.7w0.9_0.1)[Seed_49]"},
        {"name": "TverskyHD(a0.3_b0.7w0.9_0.1)[Seed_50]"},

        # {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]_new"},
    ]
    # ==============================
    
    # === 开始评估 ===
    results = []

    for exp in experiments:
        m_path = os.path.join(r"D:\Work\Python\_MSDT\saved_results", str(exp['name']), "models\\best_model.pth")
        print(f"Path: {m_path}")
        try:
            result = evaluate_single_model(m_path, test_loader, device)
            
            # 一行显示一个指标（控制台输出）
            print(f"Experiment: {exp['name']}")
            print(f"Val Loss: {result['Val_Loss']:.4f}")
            print(f"Dice (DSC): {result['DSC']:.4f}")
            print(f"IoU: {result['IoU']:.4f}")
            print(f"Accuracy: {result['Accuracy']:.4f}")
            print(f"Recall: {result['Recall']:.4f}")
            print(f"Precision: {result['Precision']:.4f}")
            pr_auc_str = f"{result['PR_AUC']:.4f}" if not np.isnan(result['PR_AUC']) else "N/A"
            print(f"PR-AUC: {pr_auc_str}")
            print(f"Average Surface Distance (ASD): {result['ASD']:.4f}")
            print(f"Boundary F1 Score: {result['Boundary_F1']:.4f}")
            print("-" * 50)
            
            results.append({'name': exp['name'], 'metrics': result})
            
        except Exception as e:
            print(f"Experiment: {exp['name']}")
            print(f"ERROR: {str(e)}")
            print("-" * 50)

    # === 保存完整结果到文件（一行一个指标）===
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Comprehensive Evaluation Results\n")
        f.write("="*60 + "\n")
        
        for result_dict in results:
            exp_name = result_dict['name']
            metrics = result_dict['metrics']
            
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Val Loss: {metrics['Val_Loss']:.4f}\n")
            f.write(f"Dice (DSC): {metrics['DSC']:.4f}\n")
            f.write(f"IoU: {metrics['IoU']:.4f}\n")
            f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
            f.write(f"Recall: {metrics['Recall']:.4f}\n")
            f.write(f"Precision: {metrics['Precision']:.4f}\n")
            pr_auc_str = f"{metrics['PR_AUC']:.4f}" if not np.isnan(metrics['PR_AUC']) else "N/A"
            f.write(f"PR-AUC: {pr_auc_str}\n")
            f.write(f"Average Surface Distance (ASD): {metrics['ASD']:.4f}\n")
            f.write(f"Boundary F1 Score: {metrics['Boundary_F1']:.4f}\n")
            f.write("-" * 60 + "\n")
            
    print(f"\n✅ Complete results saved to: {output_file}")