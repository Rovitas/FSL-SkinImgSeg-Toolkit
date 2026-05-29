# -*- coding: utf-8 -*-
import os
import datetime
import csv  # 新增 csv 库
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


# ================= 核心控制入口 =================
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ================= 1. 路径与数据配置 =================
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    RESULTS_BASE_DIR = r"D:\Work\Python\_MSDT\saved_results"
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    test_dataset = MYDataset(base_dir=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    print(f"Test set size: {len(test_dataset)}\n")
    
    # ================= 2. CSV 输出模式配置 =================
    # True  => 续写到已有的 CSV 文件尾部（比如 Losses_Comparsion.csv）
    # False => 每次运行都新建一个带时间戳的隔离文件
    APPEND_MODE = True  
    
    # 当 APPEND_MODE = True 时生效的目标文件路径
    TARGET_CSV_PATH = r"D:\Work\Python\_MSDT\calcuMetrics\Losses_Comparsion.csv" 
    
    # ================= 3. 待评估模型列表 (精确匹配) =================
    experiments = [
        # {"name": "Adam_wd_1e-05[Seed_48]"},
        # {"name": "AdamW_wd_1e-05[Seed_48]"},
        # {"name": "DiceLoss[Seed_48]"},
        # {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]"},
        # {"name": "BCEDiceLoss[Seed_48]_Aug"},
        # {"name": "BCEDiceLoss[Seed_49]_Aug"},
        # {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]_Aug"},
        # {"name": "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_49]_Aug"},
        {"name": "TverskyHD(a0.3_b0.7w0.6_0.4)[Seed_49]"},
        {"name": "TverskyHD(a0.3_b0.7w0.7_0.3)[Seed_49]"},
        {"name": "TverskyHD(a0.3_b0.7w0.6_0.4)[Seed_50]"},
        {"name": "TverskyHD(a0.3_b0.7w0.7_0.3)[Seed_50]"},
        # 在这里按需增减你想算的模型文件夹名字...
    ]
    
    # ================= 4. 初始化 CSV 写入管线 =================
    if APPEND_MODE:
        output_csv = TARGET_CSV_PATH
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"evaluation_results_{timestamp}.csv"
        
    # 判断是否需要写入表头 (如果文件不存在，或者文件是个空文件)
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    
    # 严格对应你画图代码里的指标名称
    fieldnames = [
        'Experiment', 'Val Loss', 'Dice (DSC)', 'IoU', 
        'Accuracy', 'Recall', 'Precision', 'PR-AUC', 
        'Average Surface Distance (ASD)', 'Boundary F1 Score'
    ]
    
    print(f"🚀 开始评估，定量结果将直接写入: {output_csv}")
    print("-" * 80)
    
    # 打开 CSV 文件准备写入 (使用 'a' 追加模式，utf-8-sig 防止 Excel 打开中文乱码)
    with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
            
        # ================= 5. 开始批量推理与计算 =================
        for exp in experiments:
            exp_name = exp['name']
            print(f"⏳ Evaluating: {exp_name} ...")
            
            # 采用你之前指定的精准路径匹配逻辑
            m_path = os.path.join(RESULTS_BASE_DIR, exp_name, "models", "best_model.pth")
            
            if not os.path.exists(m_path):
                print(f"❌ 找不到模型权重文件: {m_path}\n")
                continue
                
            try:
                # 调用你原有的计算逻辑
                metrics = evaluate_single_model(m_path, test_loader, DEVICE)
                
                # 格式化数值 (处理可能的 NaN 值)
                pr_auc_str = f"{metrics['PR_AUC']:.4f}" if not np.isnan(metrics['PR_AUC']) else "N/A"
                
                # 将字典组装成 CSV 的一行
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
                
                # 写入当前行并立即强制保存到硬盘 (flush)
                writer.writerow(row_dict)
                f.flush()
                
                print(f"✅ 写入成功! DSC: {row_dict['Dice (DSC)']} | IoU: {row_dict['IoU']} | ASD: {row_dict['Average Surface Distance (ASD)']}")
                print("-" * 60)
                
            except Exception as e:
                print(f"❌ 评估出错 {exp_name}: {str(e)}\n")

    print(f"🎉 所有模型定量评估完成！数据已安全归档至: {output_csv}")