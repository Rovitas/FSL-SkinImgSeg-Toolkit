# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from myDataset import MYDataset
from myModel import UNet
import matplotlib.pyplot as plt

# ================= 全局配置区 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 字体与布局控制
TITLE_FONT_SIZE = 14      # 每张子图标题的字体大小
FIG_HEIGHT = 5            # 画布的高度 (单位：英寸)
WIDTH_PER_COL = 3.5       # 每一列图像占用的宽度 (单位：英寸)
# ============================================

# 模块 1: 模型加载引擎 (精确匹配)
def load_selected_models(experiments_dict, results_base_dir):
    """
    根据配置字典中的精确文件夹名称，直接加载对应的模型权重
    """
    models = {}
    
    for display_name, exact_folder_name in experiments_dict.items():
        # 精确拼接路径
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

# 模块 2: 定性绘图引擎
def plot_qualitative_comparison(original_img, ground_truth, preds_dict, save_path, 
                                title_fs=TITLE_FONT_SIZE, fig_h=FIG_HEIGHT, w_per_col=WIDTH_PER_COL):
    """
    负责单张对比图的排版与保存
    """
    n_cols = len(preds_dict) + 2
    total_width = n_cols * w_per_col
    
    fig, axes = plt.subplots(1, n_cols, figsize=(total_width, fig_h))
    
    items = [
        ('Original Image', np.clip(original_img, 0, 1), None),
        ('Ground Truth', ground_truth, 'gray')
    ]
    items.extend([(f'{name}', pred, 'gray') for name, pred in preds_dict.items()])
    
    for i, (title, img, cmap) in enumerate(items):
        ax = axes[i]
        ax.set_title(title, fontsize=title_fs, fontweight='bold', pad=12)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 模块 3: 业务流水线 (新增过滤控制)
def run_qualitative_pipeline(test_loader, models_dict, save_dir, max_samples=None, target_ids=None):
    """
    遍历测试集并分发绘图任务
    :param max_samples: int, 限制最大输出数量，如 5 表示只输出前 5 张。为 None 则不限制。
    :param target_ids: list, 指定要输出的样本索引或名称，如 [12, 15, 'img_88']。如果提供，将忽略 max_samples。
    """
    if not models_dict:
        print("❌ 错误: 没有成功加载任何可用的模型。")
        return

    os.makedirs(save_dir, exist_ok=True)
    processed_count = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            # 兼容获取样本的标识 (如果 dataset 没有特殊属性，就默认为 img_idx)
            # 这里先尝试拿 dataset 中的 ID，拿不到就用纯数字的字符串代替
            sample_name = getattr(test_loader.dataset, 'sample_id', f'img_{idx}')
            
            # --- 核心过滤逻辑 ---
            if target_ids is not None:
                # 检查当前 idx 或名字是否在你指定的列表里
                if (idx not in target_ids) and (sample_name not in target_ids):
                    continue
            else:
                # 如果没有指定目标列表，但设置了最大数量，超额则退出
                if max_samples is not None and processed_count >= max_samples:
                    print(f"\n## 已达到设定的最大输出数量 ({max_samples}张)，终止生成。")
                    break

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 批量获取所有模型对该图的预测结果
            current_preds = {}
            for name, model in models_dict.items():
                out = model(images)[0].cpu().numpy().squeeze()
                current_preds[name] = (out >= 0.5).astype(np.float32)
            
            # 转换原图和标签为 Numpy 格式
            raw_img = images[0].cpu().numpy().transpose(1, 2, 0)
            gt_mask = labels[0].cpu().numpy().squeeze()
            
            save_name = os.path.join(save_dir, f'comparison_{sample_name}.png')
            
            # 调用绘图引擎
            plot_qualitative_comparison(raw_img, gt_mask, current_preds, save_name)
            
            print(f"## 生成成功: comparison_{sample_name}.png")
            processed_count += 1

# ================= 核心控制入口 =================
if __name__ == "__main__":
    # --- 1. 路径设置 ---
    TEST_DIR = r"d:\Work\Python\_MSDT\images_split\test"
    RESULTS_ROOT = r"D:\Work\Python\_MSDT\saved_results"
    SAVE_FOLDER = r"D:\Work\Python\_MSDT\compare_img\Losses_Comparison"
    
    # --- 2. 在这里配置你需要对比的模型 (精确匹配) ---
    EXPERIMENTS_TO_COMPARE = {
        'Baseline': "Adam_wd_1e-05[Seed_48]",          
        'BCE': "AdamW_wd_1e-05[Seed_48]",              
        # 'Dice': "DiceLoss[Seed_48]",
        # 'Focal': "FocalLoss(a0.25_g2)[Seed_48]",
        # 'Tversky': "TverskyLoss(a0.3_b0.7)[Seed_48]",
        # 'TverskyHD55': "TverskyHD(a0.3_b0.7w0.5_0.5)[Seed_48]",
        'TverskyHD82': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]",
        'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)[Seed_48]"
    }
    
    # --- 3. 初始化数据加载 ---
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_set = MYDataset(base_dir=TEST_DIR, transform=transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    # --- 4. 生成数量与范围控制 ---
    # 控制只画前几张图 (如设为 None 则画全部)
    MAX_OUTPUT_IMAGES = 5  
    
    # 控制指定画哪几张特定的图 (填入列表如 [0, 5, 12] 或 ['img_10'])
    # 注意：如果不为 None，程序将无视上面的 MAX_OUTPUT_IMAGES 参数，只画列表中指定的图
    TARGET_IMAGE_IDS = None  
    
    # --- 5. 运行流水线 ---
    print("正在加载模型Loading models...")
    active_models = load_selected_models(EXPERIMENTS_TO_COMPARE, RESULTS_ROOT)
    
    print(f"\n## 开始定性对比分析，保存路径: {SAVE_FOLDER}")
    run_qualitative_pipeline(
        test_loader=loader, 
        models_dict=active_models, 
        save_dir=SAVE_FOLDER,
        max_samples=MAX_OUTPUT_IMAGES,
        target_ids=TARGET_IMAGE_IDS
    )
    
    print("\n## 定性分析完成！请检查输出文件夹：")
    print(f"{SAVE_FOLDER}")