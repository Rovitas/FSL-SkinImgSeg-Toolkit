# -*- coding: utf-8 -*-
import csv
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import auc, precision_recall_curve
from torch.utils.data import DataLoader

code_root = Path(__file__).resolve().parents[1]
sys.path.append(str(code_root))

from train.my_dataset import MYDataset
from train.my_model import UNet

# ================= 核心计算逻辑 =================
def get_boundary(binary_mask):
    return binary_mask - binary_erosion(binary_mask)

def average_surface_distance(pred, target):
    N = pred.shape[0]
    asd_distances = []
    for b in range(N):
        pred_b = (pred[b, 0] > 0.5).cpu().numpy().astype(np.uint8)
        target_b = (target[b, 0] > 0.5).cpu().numpy().astype(np.uint8)
        pred_bound, target_bound = get_boundary(pred_b), get_boundary(target_b)
        
        if pred_bound.sum() == 0 and target_bound.sum() == 0:
            asd_distances.append(0.0); continue
        elif pred_bound.sum() == 0 or target_bound.sum() == 0:
            asd_distances.append(np.sqrt(pred_b.shape[0]**2 + pred_b.shape[1]**2)); continue
            
        target_dist = distance_transform_edt(1 - target_bound)
        pred_dist = distance_transform_edt(1 - pred_bound)
        asd = (target_dist[pred_bound.astype(bool)].mean() + pred_dist[target_bound.astype(bool)].mean()) / 2.0
        asd_distances.append(asd)
    return np.mean(asd_distances)

def boundary_f1_score(pred, target, threshold=0.5):
    N, bf_scores = pred.shape[0], []
    for b in range(N):
        pred_b = (pred[b, 0] > threshold).cpu().numpy().astype(np.uint8)
        target_b = (target[b, 0] > threshold).cpu().numpy().astype(np.uint8)
        p_bound, t_bound = get_boundary(pred_b), get_boundary(target_b)
        
        if p_bound.sum() == 0 and t_bound.sum() == 0: bf_scores.append(1.0); continue
        elif p_bound.sum() == 0 or t_bound.sum() == 0: bf_scores.append(0.0); continue
        
        intersection = np.logical_and(p_bound, t_bound).sum()
        precision = intersection / p_bound.sum() if p_bound.sum() > 0 else 0
        recall = intersection / t_bound.sum() if t_bound.sum() > 0 else 0
        bf_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return np.mean(bf_scores)

def calculate_metrics(pred, target, eps=1e-6):
    probs = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    pred_binary, target_binary = (probs >= 0.5).float(), (target >= 0.5).float()
    pred_flat, target_flat = pred_binary.flatten(), target_binary.flatten()

    TP = (pred_flat * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()

    try:
        p_curve, r_curve, _ = precision_recall_curve(target_flat.cpu().numpy(), probs.flatten().cpu().numpy())
        pr_auc = auc(r_curve, p_curve)
    except: pr_auc = float('nan')

    return {
        'DSC': ((2. * TP + eps) / (pred_flat.sum() + target_flat.sum() + eps)).item(),
        'IoU': ((TP + eps) / (pred_flat.sum() + target_flat.sum() - TP + eps)).item(),
        'Accuracy': ((TP + TN) / (TP + TN + FP + FN + eps)).item(),
        'Recall': (TP / (TP + FN + eps)).item(),
        'Precision': (TP / (TP + FP + eps)).item(),
        'PR_AUC': pr_auc,
        'ASD': average_surface_distance(pred_binary, target_binary),
        'Boundary_F1': boundary_f1_score(pred_binary, target_binary)
    }

def evaluate_single_model(model_path, val_loader, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_samples = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            total_samples += inputs.size(0)
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    metrics = calculate_metrics(torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0))
    metrics['Val_Loss'] = total_loss / total_samples
    return metrics


# ================= 业务流水线 =================
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEST_DIR = os.path.join(PROJECT_ROOT, "images_split", "test")
    RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, "saved_results")
    
    # 待评估模型列表 (要求填写精确的文件夹名)
    EVAL_EXPERIMENTS = [
        "TverskyHD(a0.3_b0.7w0.6_0.4)[Seed_49]",
        "TverskyHD(a0.3_b0.7w0.7_0.3)[Seed_49]",
        "TverskyHD(a0.3_b0.7w0.6_0.4)[Seed_50]",
        "TverskyHD(a0.3_b0.7w0.7_0.3)[Seed_50]"
    ]
    
    APPEND_MODE = True  
    TARGET_CSV_PATH = os.path.join(PROJECT_ROOT, "calcuMetrics", "Losses_Comparsion.csv")
    
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_dataset = MYDataset(base_dir=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    if APPEND_MODE:
        output_csv = TARGET_CSV_PATH
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(PROJECT_ROOT, "calcuMetrics", f"evaluation_results_{timestamp}.csv")
        
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    fieldnames = [
        'Experiment', 'Val Loss', 'Dice (DSC)', 'IoU', 
        'Accuracy', 'Recall', 'Precision', 'PR-AUC', 
        'Average Surface Distance (ASD)', 'Boundary F1 Score'
    ]
    
    print(f"🚀 开始评估，定量结果将直接写入: {output_csv}\n" + "=" * 60)
    
    with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        for exp_name in EVAL_EXPERIMENTS:
            m_path = os.path.join(RESULTS_BASE_DIR, exp_name, "models", "best_model.pth")
            
            if not os.path.exists(m_path):
                print(f"❌ 找不到模型文件: {m_path}\n" + "-" * 60)
                continue
                
            try:
                metrics = evaluate_single_model(m_path, test_loader, DEVICE)
                pr_auc_str = f"{metrics['PR_AUC']:.4f}" if not np.isnan(metrics['PR_AUC']) else "N/A"
                
                row_dict = {
                    'Experiment': exp_name,
                    'Val Loss': f"{metrics['Val_Loss']:.4f}",
                    'Dice (DSC)': f"{metrics['DSC']:.4f}",
                    'IoU': f"{metrics['IoU']:.4f}",
                    'Accuracy': f"{metrics['Accuracy']:.4f}",
                    'Recall': f"{metrics['Recall']:.4f}",
                    'Precision': f"{metrics['Precision']:.4f}",
                    'PR-AUC': pr_auc_str,
                    'Average Surface Distance (ASD)': f"{metrics['ASD']:.4f}",
                    'Boundary F1 Score': f"{metrics['Boundary_F1']:.4f}"
                }
                
                writer.writerow(row_dict)
                f.flush()
                
                exp_str = (
                    f"Experiment: {exp_name}\n"
                    f"Val Loss: {metrics['Val_Loss']:.4f} | Dice (DSC): {metrics['DSC']:.4f} | IoU: {metrics['IoU']:.4f}\n"
                    f"Accuracy: {metrics['Accuracy']:.4f} | Recall: {metrics['Recall']:.4f} | Precision: {metrics['Precision']:.4f}\n"
                    f"PR-AUC: {pr_auc_str} | ASD: {metrics['ASD']:.4f} | Bound-F1: {metrics['Boundary_F1']:.4f}\n"
                    + "-" * 60
                )
                print(exp_str)
                
            except Exception as e:
                print(f"❌ 评估出错 {exp_name}: {str(e)}\n" + "-" * 60)

    print(f"\n🎉 评估完成！数据已安全归档至: {output_csv}")