# -*- coding: utf-8 -*-
import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from scipy.ndimage import binary_erosion, distance_transform_edt
import warnings
warnings.filterwarnings("ignore")

def get_boundary(binary_mask):
    return binary_mask - binary_erosion(binary_mask)

def average_surface_distance(pred, target):
    """保持原计算逻辑完全不变"""
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
    """保持原计算逻辑完全不变"""
    N, bf_scores = pred.shape[0], []
    for b in range(N):
        pred_b = (pred[b, 0] > threshold).cpu().numpy().astype(np.uint8)
        target_b = (target[b, 0] > threshold).cpu().numpy().astype(np.uint8)
        p_bound, t_bound = get_boundary(pred_b), get_boundary(target_b)
        
        if p_bound.sum() == 0 and t_bound.sum() == 0: bf_scores.append(1.0); continue
        elif p_bound.sum() == 0 or t_bound.sum() == 0: bf_scores.append(0.0); continue
        
        intersection = np.logical_and(p_bound, t_bound).sum()
        precision = intersection / p_bound.sum()
        recall = intersection / t_bound.sum()
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

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    
    test_loader = DataLoader(MYDataset(base_dir=TEST_DIR, transform=transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor()
    ])), batch_size=4, shuffle=False, num_workers=2)

    experiments = [
        {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_49]"},
        {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_50]"},
        # 你的其余模型配置保持不变...
    ]
    
    results = []
    output_str = "Comprehensive Evaluation Results\n" + "="*60 + "\n"
    
    # 完美复刻原有的输出排版
    for exp in experiments:
        m_path = os.path.join(r"D:\Work\Python\_MSDT\saved_results", str(exp['name']), "models", "best_model.pth")
        try:
            metrics = evaluate_single_model(m_path, test_loader, DEVICE)
            pr_auc_str = f"{metrics['PR_AUC']:.4f}" if not np.isnan(metrics['PR_AUC']) else "N/A"
            
            exp_str = (
                f"Experiment: {exp['name']}\n"
                f"Val Loss: {metrics['Val_Loss']:.4f}\n"
                f"Dice (DSC): {metrics['DSC']:.4f}\n"
                f"IoU: {metrics['IoU']:.4f}\n"
                f"Accuracy: {metrics['Accuracy']:.4f}\n"
                f"Recall: {metrics['Recall']:.4f}\n"
                f"Precision: {metrics['Precision']:.4f}\n"
                f"PR-AUC: {pr_auc_str}\n"
                f"Average Surface Distance (ASD): {metrics['ASD']:.4f}\n"
                f"Boundary F1 Score: {metrics['Boundary_F1']:.4f}\n"
                + "-" * 60 + "\n"
            )
            print(exp_str, end="") # 打印到控制台
            output_str += exp_str  # 累加到文件文本
            
        except Exception as e:
            err_str = f"Experiment: {exp['name']}\nERROR: {str(e)}\n" + "-" * 60 + "\n"
            print(err_str, end="")
            output_str += err_str

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_str)
        
    print(f"\n✅ Complete results saved to: {output_file}")