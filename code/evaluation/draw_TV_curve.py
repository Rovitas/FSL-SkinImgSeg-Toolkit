import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def smooth_loss_ewma(losses, alpha=0.3):
    """指数加权移动平均（EWMA）平滑"""
    smoothed = [losses[0]]
    for i in range(1, len(losses)):
        smoothed.append(alpha * losses[i] + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def parse_log_file(log_path):
    """解析 training.log 文件"""
    epochs, train_losses, val_losses = [], [], []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            losses = re.search(r'→ Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
            if losses:
                epochs.append(len(train_losses) + 1)
                train_losses.append(float(losses.group(1)))
                val_losses.append(float(losses.group(2)))
    return epochs, train_losses, val_losses

def draw_curve(dicts, logs_dir, wd_colors=None, smooth_method='ewma', train_a=0.3, val_a=0.3, show_std=True, save_path=None, show_plot=False):

    if wd_colors is None:
        wd_colors = {}
        
    data_dict = {'Adam': {}, 'AdamW': {}}
    
    for exp_name, log_paths in dicts.items():
        if exp_name.startswith('AdamW'):
            opt = 'AdamW'
        elif 'Adam' in exp_name:
            opt = 'Adam'
        else:
            continue

        wd_match = re.search(r'WD[=:]([0-9e\-.]+)', exp_name)
        wd_key = wd_match.group(1) if wd_match else '0'

        if wd_key not in data_dict[opt]:
            data_dict[opt][wd_key] = {'train': [], 'val': []}

        for log_file in log_paths:
            # [修改] 使用传入的 logs_dir 替代硬编码绝对路径
            full_path = os.path.join(logs_dir, log_file)
            if not os.path.exists(full_path):
                print(f"⚠️ 路径不存在，已跳过: {full_path}")
                continue
                
            eps, train_l, val_l = parse_log_file(full_path)
            if train_l:
                data_dict[opt][wd_key]['train'].append(train_l)
            if val_l:
                data_dict[opt][wd_key]['val'].append(val_l)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)

    def plot_subplot(ax, opt_data, loss_type, title):
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('Loss', fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.8)

        for wd, losses_dict in opt_data.items():
            all_losses = losses_dict[loss_type]
            if not all_losses:
                continue

            max_epochs = max(len(l) for l in all_losses)
            padded_losses = []
            for l in all_losses:
                if len(l) < max_epochs:
                    padded_losses.append(l + [l[-1]] * (max_epochs - len(l)))
                else:
                    padded_losses.append(l)

            avg_loss = np.mean(padded_losses, axis=0)
            std_loss = np.std(padded_losses, axis=0)
            epochs = np.arange(1, max_epochs + 1)

            if smooth_method == 'ewma':
                alpha = train_a if loss_type == 'train' else val_a
                smoothed_mean = smooth_loss_ewma(avg_loss, alpha=alpha)
                smoothed_upper = smooth_loss_ewma(avg_loss + std_loss, alpha=alpha)
                smoothed_lower = smooth_loss_ewma(avg_loss - std_loss, alpha=alpha)
            else:
                smoothed_mean = avg_loss
                smoothed_upper = avg_loss + std_loss
                smoothed_lower = avg_loss - std_loss

            color = wd_colors.get(wd, '#000000') 
            label = f'WD={wd}'
            ax.plot(epochs, smoothed_mean, label=label, color=color, linewidth=2.5, alpha=0.9)
            
            if show_std and len(all_losses) > 1:
                ax.fill_between(epochs, smoothed_lower, smoothed_upper, color=color, alpha=0.15, edgecolor='none')
            
            ax.tick_params(axis='both', labelsize=16)

        ax.legend(title='Weight Decay', fontsize=19, title_fontsize=19)

    # --- 绘制四张子图 ---
    plot_subplot(axes[0, 0], data_dict['Adam'], 'train', 'Adam: Training Loss')
    plot_subplot(axes[0, 1], data_dict['AdamW'], 'train', 'AdamW: Training Loss')
    plot_subplot(axes[1, 0], data_dict['Adam'], 'val', 'Adam: Validation Loss')
    plot_subplot(axes[1, 1], data_dict['AdamW'], 'val', 'AdamW: Validation Loss')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Saved plot to: {save_path}")
        
    if show_plot: 
        plt.show()
        
    plt.close()

if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs_summary")
    IMG_PATH = os.path.join(PROJECT_ROOT, "Curve.png")
    SHOW_FLAG = False

    experiments = {
        "Baseline (Adam + WD=0)": [
            r"Adam_wd_0[Seed_38]_Baseline.log",
            r"Adam_wd_0[Seed_39]_Baseline.log",
            r"Adam_wd_0[Seed_40]_Baseline.log",
            r"Adam_wd_0[Seed_48]_Baseline.log",
        ],
        "Adam + WD=1e-5": [
            r"Adam_wd_1e-05[Seed_48].log",
            r"Adam_wd_1e-05[Seed_50].log",
        ],
        "Adam + WD=5e-5": [
            r"Adam_wd_5e-05[Seed_48].log",
            r"Adam_wd_5e-05[Seed_49].log",
        ],
        "Adam + WD=1e-4": [
            r"Adam_wd_1e-04[Seed_48].log",
            r"Adam_wd_1e-04[Seed_49].log",
        ],
        "AdamW + WD=0": [
            r"AdamW_wd_0[Seed_48].log",
            r"AdamW_wd_0[Seed_49].log",
            r"AdamW_wd_0[Seed_50].log",
        ],
        "AdamW + WD=1e-5": [
            r"AdamW_wd_1e-05[Seed_48].log",
            r"AdamW_wd_1e-05[Seed_49].log",
            r"AdamW_wd_1e-05[Seed_50].log",
        ],
        "AdamW + WD=5e-5": [
            r"AdamW_wd_5e-05[Seed_48].log",
            r"AdamW_wd_5e-05[Seed_49].log",
            r"AdamW_wd_5e-05[Seed_50].log",
        ],
        "AdamW + WD=1e-4": [
            r"AdamW_wd_1e-04[Seed_48].log",
            r"AdamW_wd_1e-04[Seed_49].log",
            r"AdamW_wd_1e-04[Seed_50].log",
        ]
    }
    
    # 灵活分离出的颜色映射配置
    wd_colors_map = {
        '0': '#e41a1c',
        '1e-5': '#377eb8',
        '5e-5': '#4daf4a',
        '1e-4': '#984ea3'
    }

    
    draw_curve(
        dicts=experiments, 
        logs_dir=LOGS_DIR,
        wd_colors=wd_colors_map,
        smooth_method='ewma', 
        train_a=0.3, 
        val_a=0.1, 
        show_std=True,
        save_path=IMG_PATH, 
        show_plot=SHOW_FLAG
    )