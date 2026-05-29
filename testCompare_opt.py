# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================= 全局配置区 (统一管理样式) =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 字体与布局控制
TITLE_FONT_SIZE = 16      # 每张子图标题的字体大小
CELL_SIZE = 3.5            # 【修改点1】: 统一用一个参数控制每个正方形格子的边长，消除长宽比冲突带来的巨大空白

# GT 轮廓线与显示控制
GT_CONTOUR_COLOR = '#00FF00'  
GT_CONTOUR_LINEWIDTH = 4      
GT_CONTOUR_ALPHA = 0.7        
SHOW_GT_SUBPLOT = False       # 【新增开关】: 设为 False 则隐藏单独的 Ground Truth 格子，只在其他图上显示绿线
# ============================================================

#修改RC参数，使其支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模块 1: 模型加载引擎 (精确匹配)
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

# 模块 2: 定性绘图引擎 (增加 GT 显示开关)
def plot_qualitative_comparison(original_img, ground_truth, preds_dict, save_path, 
                                n_rows, n_cols, show_gt=SHOW_GT_SUBPLOT):
    # 【核心修改】：根据开关决定是否把 GT 单独作为一个图放进列表中
    # items = [('原始图像', np.clip(original_img, 0, 1), None)]     #中文
    items = [('Original Image', np.clip(original_img, 0, 1), None)]     #English
    
    if show_gt:
        items.append(('Ground Truth', ground_truth, 'gray'))
        
    items.extend([(f'{name}', pred, 'gray') for name, pred in preds_dict.items()])
    
    if len(items) > n_rows * n_cols:
        print(f"⚠️ 警告: 要画的图({len(items)}张)多于网格数({n_rows}x{n_cols})。超出的部分将被忽略！")
        items = items[:n_rows * n_cols]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * CELL_SIZE, n_rows * CELL_SIZE + 1.8))
    axes = np.atleast_1d(axes).flatten()
    
    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < len(items):
            title, img, cmap = items[i]
            ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=10)
            ax.imshow(img, cmap=cmap)
            
            # 画出 GT 的边缘轮廓 (即使上面的单独 GT 图隐藏了，这里的 ground_truth 数据依然存在并能正常画线)
            if title != 'Ground Truth': 
                ax.contour(ground_truth, levels=[0.5], colors=[GT_CONTOUR_COLOR], 
                           linewidths=GT_CONTOUR_LINEWIDTH, alpha=GT_CONTOUR_ALPHA)
            ax.axis('off')
        else:
            ax.axis('off')
            
    # gt_patch = mpatches.Patch(color=GT_CONTOUR_COLOR, label='专家标注的真实边界GT的投影') #中文
    gt_patch = mpatches.Patch(color=GT_CONTOUR_COLOR, label='Ground Truth Contour')     #English

    # 【核心修复】：使用强力参数压缩子图之间的水平 (w_pad) 和垂直 (h_pad) 间距
    plt.tight_layout(w_pad=0.2, h_pad=1.0) 
    
    # 给底部的图例额外留一点空间，防止被切掉
    fig.subplots_adjust(bottom=0.05) 
    fig.legend(handles=[gt_patch], loc='lower center', ncol=1, bbox_to_anchor=(0.5, 0.03), fontsize=14)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 模块 3: 业务流水线
def run_qualitative_pipeline(test_loader, models_dict, save_dir, max_samples=None, target_ids=None,
                             grid_rows=1, grid_cols=5):
    if not models_dict:
        print("❌ 错误: 没有成功加载任何可用的模型。")
        return

    os.makedirs(save_dir, exist_ok=True)
    processed_count = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            sample_name = getattr(test_loader.dataset, 'sample_id', f'img_{idx}')
            
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

def main(mode, test_img, result_root, save_folder, max_amount=None, choose_img_id=None):
    if mode.lower() == 'optimizer':
        EXPERIMENTS_TO_COMPARE = {
            'Adam': "Adam_wd_1e-05[Seed_50]",          
            'AdamW': "AdamW_wd_1e-05[Seed_50]",      
            # 'Adam优化器': "Adam_wd_1e-05[Seed_50]",          
            # 'AdamW优化器': "AdamW_wd_1e-05[Seed_50]",        
        }
        save_folder = os.path.join(save_folder, 'Adam-AdamW')  
        LAYOUT_ROWS = 1  
        LAYOUT_COLS = 3 
    elif IMG_MODE.lower() == 'loss':
        EXPERIMENTS_TO_COMPARE = {   
            'BCE': "AdamW_wd_1e-05[Seed_50]",                  
            'Dice': "DiceLoss[Seed_49]",
            'Focal': "FocalLoss(a0.25_g2)[Seed_49]",
            'Tversky': "TverskyLoss(a0.3_b0.7)[Seed_50]",
            # 'TverskyHD55': "TverskyHD(a0.3_b0.7w0.5_0.5)[Seed_48]",
            'TverskyHD': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_50]",       
        }
        save_folder = os.path.join(save_folder, 'Losses_Comparison')
        LAYOUT_ROWS = 2  
        LAYOUT_COLS = 3 
    elif mode.lower() == 'tverskyhd_pre':
        EXPERIMENTS_TO_COMPARE = {   
            'TverskyHD 5:5': "TverskyHD(a0.3_b0.7w0.5_0.5)[Seed_49]", 
            'TverskyHD 6:4': "TverskyHD(a0.3_b0.7w0.6_0.4)[Seed_50]", 
            'TverskyHD 7:3': "TverskyHD(a0.3_b0.7w0.7_0.3)[Seed_49]", 
            'TverskyHD 8:2': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_50]", 
            'TverskyHD 9:1': "TverskyHD(a0.3_b0.7w0.9_0.1)[Seed_50]",    
        }
        save_folder = os.path.join(save_folder, 'TverskyHD_Pre')
        LAYOUT_ROWS = 2  
        LAYOUT_COLS = 3 
        
    elif mode.lower() == 'xiaorong':
        EXPERIMENTS_TO_COMPARE = {
            # '基线方法（BCE+Adam）': "Adam_wd_1e-05[Seed_50]", 
            'Baseline(BCE+Adam)': "Adam_wd_1e-05[Seed_50]",         
            'BCE+AdamW': "AdamW_wd_1e-05[Seed_50]", 
            'BCE+AdamW+Aug': "BCEDiceLoss[Seed_48]_Aug", 
            'TverskyHD+Adam':'TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]_Single',  
            'TverskyHD+AdamW': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_50]", 
            'Full(TverskyHD+AdamW+Aug)': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_49]_Aug",
        }
        save_folder = os.path.join(save_folder, 'XIAORONG')
        LAYOUT_ROWS = 2  
        LAYOUT_COLS = 4 
    else:
        print(f"⚠️ Unknown mode: {mode}")   

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_set = MYDataset(base_dir=test_img, transform=transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    if choose_img_id is None: choose_img_id = ['IMD284', 'IMD410', 'IMD424']
    
    print("⏳ 正在根据配置精确加载模型...")
    active_models = load_selected_models(EXPERIMENTS_TO_COMPARE, result_root)
    
    print(f"\n🚀 开始定性对比分析")
    run_qualitative_pipeline(
        test_loader=loader, 
        models_dict=active_models, 
        save_dir=save_folder,
        max_samples=max_amount,   
        target_ids=choose_img_id,
        grid_rows=LAYOUT_ROWS,          
        grid_cols=LAYOUT_COLS           
    )
    print(f"\n🎉 定性分析完成！请检查输出文件夹:{SAVE_FOLDER}\n")


if __name__ == "__main__":
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    RESULTS_ROOT = r"D:\Work\Python\_MSDT\saved_results"
    SAVE_FOLDER = r"D:\Work\Python\_MSDT\compare_img"
    MAX_OUTPUT_IMAGES = None 
    TARGET_IMAGE_IDS = None
    
    IMG_MODE = 'optimizer' 
    # IMG_MODE = 'loss'
    # IMG_MODE = 'xiaorong'  
    # IMG_MODE = 'tverskyhd_pre'

    main(mode=IMG_MODE, test_img=TEST_DIR, result_root=RESULTS_ROOT, save_folder=SAVE_FOLDER, max_amount=MAX_OUTPUT_IMAGES, choose_img_id=TARGET_IMAGE_IDS)