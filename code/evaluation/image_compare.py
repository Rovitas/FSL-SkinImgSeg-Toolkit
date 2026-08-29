# -*- coding: utf-8 -*-
import math
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

code_root = Path(__file__).resolve().parents[1]
sys.path.append(str(code_root))

from train.my_dataset import MYDataset
from train.my_model import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TITLE_FONT_SIZE = 16      
CELL_SIZE = 3.5            

GT_CONTOUR_COLOR = '#00FF00'  
GT_CONTOUR_LINEWIDTH = 4      
GT_CONTOUR_ALPHA = 0.7        
SHOW_GT_SUBPLOT = False       


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_auto_grid(n_items):
    """
    自动提取因数计算网格：
    寻找最接近平方根的因数对 (r, c)，保证 r * c == n_items 且 r <= c
    """
    r = math.floor(math.sqrt(n_items))
    while n_items % r != 0:
        r -= 1
    c = n_items // r
    return r, c


def load_selected_models(experiments_dict, results_base_dir):
    models = {}
    for display_name, exact_folder_name in experiments_dict.items():
        model_path = os.path.join(results_base_dir, exact_folder_name, "models", "best_model.pth")
        if os.path.exists(model_path):
            print(f"Loading {display_name}: {model_path}") 
            model = UNet().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True)['model_state_dict'])
            model.eval()
            models[display_name] = model
        else:
            print(f"❌ 警告: 找不到指定的模型文件: {model_path}")
    return models


def plot_qualitative_comparison(original_img, ground_truth, preds_dict, save_path, 
                                n_rows=None, n_cols=None, show_gt=SHOW_GT_SUBPLOT):
    
    items = [('Original Image', np.clip(original_img, 0, 1), None)]
    
    if show_gt:
        items.append(('Ground Truth', ground_truth, 'gray'))
        
    items.extend([(f'{name}', pred, 'gray') for name, pred in preds_dict.items()])
    
    total_items = len(items)

    if n_rows is None or n_cols is None:
        n_rows, n_cols = get_auto_grid(total_items)
    
    if total_items > n_rows * n_cols:
        print(f"⚠️ 警告: 要画的图({total_items}张)多于指定的网格数({n_rows}x{n_cols})。超出的部分将被忽略！")
        items = items[:n_rows * n_cols]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * CELL_SIZE, n_rows * CELL_SIZE + 1.8))
    axes = np.atleast_1d(axes).flatten()
    
    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < len(items):
            title, img, cmap = items[i]
            ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=10)
            ax.imshow(img, cmap=cmap)
            
            if title != 'Ground Truth': 
                ax.contour(ground_truth, levels=[0.5], colors=[GT_CONTOUR_COLOR], 
                           linewidths=GT_CONTOUR_LINEWIDTH, alpha=GT_CONTOUR_ALPHA)
            ax.axis('off')
        else:
            ax.axis('off')
            
    gt_patch = mpatches.Patch(color=GT_CONTOUR_COLOR, label='Ground Truth Contour')

    plt.tight_layout(w_pad=0.2, h_pad=1.0) 
    fig.subplots_adjust(bottom=0.05) 
    fig.legend(handles=[gt_patch], loc='lower center', ncol=1, bbox_to_anchor=(0.5, 0.03), fontsize=14)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_qualitative_pipeline(test_loader, models_dict, save_dir, grid_rows=None, grid_cols=None, 
                             max_samples=None, target_ids=None):
    if not models_dict:
        print("❌ 错误: 没有成功加载任何可用的模型。")
        return

    os.makedirs(save_dir, exist_ok=True)
    processed_count = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            sample_name = test_loader.dataset.sample_ids[idx]
            
            if target_ids is not None:
                if (idx not in target_ids) and (sample_name not in target_ids):
                    continue
            else:
                if max_samples is not None and processed_count >= max_samples:
                    print(f"\n⏹ 已达到设定的最大输出数量 ({max_samples}张)，终止生成。")
                    break

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            current_preds = {}
            for name, model in models_dict.items():
                out = model(images)[0].cpu().numpy().squeeze()
                current_preds[name] = (out >= 0.5).astype(np.float32)
            
            raw_img = images[0].cpu().numpy().transpose(1, 2, 0)
            gt_mask = labels[0].cpu().numpy().squeeze()
            
            save_name = os.path.join(save_dir, f'comparison_{sample_name}.png')
            plot_qualitative_comparison(raw_img, gt_mask, current_preds, save_name, grid_rows, grid_cols)
            
            print(f"✅ 生成成功: comparison_{sample_name}.png")
            processed_count += 1


def main(experiments_dict, test_img_dir, result_root, save_folder, 
         grid_rows=None, grid_cols=None, max_amount=None, choose_img_id=None):
    
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_set = MYDataset(base_dir=test_img_dir, transform=transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    if choose_img_id is None: 
        choose_img_id = ['IMD284', 'IMD410', 'IMD424']
    
    print("⏳ 正在根据配置精确加载模型...")
    active_models = load_selected_models(experiments_dict, result_root)
    
    print(f"\n🚀 开始定性对比分析")
    run_qualitative_pipeline(
        test_loader=loader, 
        models_dict=active_models, 
        save_dir=save_folder,
        grid_rows=grid_rows,          
        grid_cols=grid_cols,
        max_samples=max_amount,   
        target_ids=choose_img_id
    )
    print(f"\n🎉 定性分析完成！请检查输出文件夹: {save_folder}\n")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEST_DIR = os.path.join(PROJECT_ROOT, "images_split", "test")
    RESULTS_ROOT = os.path.join(PROJECT_ROOT, "saved_results")
    SAVE_BASE_FOLDER = os.path.join(PROJECT_ROOT, "compare_img")
    
    MAX_OUTPUT_IMAGES = None 
    TARGET_IMAGE_IDS = ['IMD284', 'IMD410', 'IMD424']

    EXPERIMENTS_CONFIG = {
        'optimizer': {
            'experiments': {
                'Adam': "Adam_wd_1e-05[Seed_50]",          
                'AdamW': "AdamW_wd_1e-05[Seed_50]",      
            },
            'subfolder': 'Adam-AdamW',
            'rows': 1,  
            'cols': 3 
        },
        'loss': {
            'experiments': {   
                'BCE': "AdamW_wd_1e-05[Seed_50]",                  
                'Dice': "DiceLoss[Seed_49]",
                'Focal': "FocalLoss(a0.25_g2)[Seed_49]",
                'Tversky': "TverskyLoss(a0.3_b0.7)[Seed_50]",
                'TverskyHD': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_50]",       
            },
            'subfolder': 'Losses_Comparison',
            'rows': None,
            'cols': None 
        },
        'tverskyhd_pre': {
            'experiments': {   
                'TverskyHD 5:5': "TverskyHD(a0.3_b0.7w0.5_0.5)[Seed_49]", 
                'TverskyHD 6:4': "TverskyHD(a0.3_b0.7w0.6_0.4)[Seed_50]", 
                'TverskyHD 7:3': "TverskyHD(a0.3_b0.7w0.7_0.3)[Seed_49]", 
                'TverskyHD 8:2': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_50]", 
                'TverskyHD 9:1': "TverskyHD(a0.3_b0.7w0.9_0.1)[Seed_50]",    
            },
            'subfolder': 'TverskyHD_Pre',
            'rows': 2,  
            'cols': 3 
        },
        'xiaorong': {
            'experiments': {
                'Baseline(BCE+Adam)': "Adam_wd_1e-05[Seed_50]",         
                'BCE+AdamW': "AdamW_wd_1e-05[Seed_50]", 
                'BCE+AdamW+Aug': "BCEDiceLoss[Seed_48]_Aug", 
                'TverskyHD+Adam': 'TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]_Single',  
                'TverskyHD+AdamW': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_50]", 
                'Full(TverskyHD+AdamW+Aug)': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_49]_Aug",
            },
            'subfolder': 'XIAORONG',
            'rows': 2,  
            'cols': 4 
        }
    }
    
    CURRENT_MODE = 'loss' 

    if CURRENT_MODE not in EXPERIMENTS_CONFIG:
        print(f"⚠️ 未知的实验模式: {CURRENT_MODE}")
    else:
        config = EXPERIMENTS_CONFIG[CURRENT_MODE]
        target_save_folder = os.path.join(SAVE_BASE_FOLDER, config['subfolder'])
        
        main(
            experiments_dict=config['experiments'], 
            test_img_dir=TEST_DIR, 
            result_root=RESULTS_ROOT, 
            save_folder=target_save_folder, 
            grid_rows=config['rows'],
            grid_cols=config['cols'],
            max_amount=MAX_OUTPUT_IMAGES, 
            choose_img_id=TARGET_IMAGE_IDS
        )