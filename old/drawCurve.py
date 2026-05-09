# -*- coding: utf-8 -*-
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_file(log_path):
    """
    解析单个 training.log 文件，返回 epochs, train_losses, val_losses
    """
    epochs = []
    train_losses = []
    val_losses = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配日志行：→ Train Loss: 0.428365 | Val Loss: 0.469316
            losses = re.search(r'→ Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
            if losses:
                train_loss = float(losses.group(1))
                val_loss = float(losses.group(2))
                # 当前 epoch = 已记录数量 + 1
                epoch = len(train_losses) + 1
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    return epochs, train_losses, val_losses

def draw_crave(dicts, smooth_method='ewma', alpha=0.3, save_path=None, figsize=(12, 6), show_plot=True):
     # 创建绘图
    plt.figure(figsize=figsize)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    markers = ['o', 's', '^', 'D', 'v', '*']

    for idx, (exp_name, log_paths) in enumerate(dicts.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        all_train_losses = []
        all_val_losses = []
        max_epochs = 0

        # 解析所有 log（用于计算平均值，如果有多次运行）
        for log_path in log_paths:
            if not os.path.exists(log_path):
                print(f"⚠️ Warning: {log_path} not found, skipping.")
                continue
            epochs, train_losses, val_losses = parse_log_file(log_path)
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            max_epochs = max(max_epochs, len(train_losses))

        if not all_train_losses:
            continue

        # 如果只有一个 run，直接绘制；如果有多个，可选绘制平均（此处先画单条）
        if len(all_train_losses) == 1:
            train_loss = all_train_losses[0]
            val_loss = all_val_losses[0]
            epochs = list(range(1, len(train_loss) + 1))

            # 绘制 Train Loss 和 Val Loss
            plt.plot(epochs, train_loss, label=f"{exp_name} (Train)", color=color, linestyle='-', linewidth=1.5)
            plt.plot(epochs, val_loss, label=f"{exp_name} (Val)", color=color, linestyle='--', linewidth=1.5)
            # 标记关键点
            # for i in range(0, len(epochs), 10):  # 每10个epoch标一次
            #     plt.text(epochs[i], train_loss[i], f'{train_loss[i]:.3f}', fontsize=7, ha='center', va='bottom', color=color)
            #     plt.text(epochs[i], val_loss[i], f'{val_loss[i]:.3f}', fontsize=7, ha='center', va='top', color=color)

        else:
            # 多次运行：计算平均（可选）
            import numpy as np
            avg_train = np.mean([l + [l[-1]]*(max_epochs-len(l)) for l in all_train_losses], axis=0)
            avg_val = np.mean([l + [l[-1]]*(max_epochs-len(l)) for l in all_val_losses], axis=0)
            epochs = list(range(1, max_epochs+1))
            plt.plot(epochs, avg_train, label=f"{exp_name} (Train, avg)", color=color, linestyle='-')
            # plt.plot(epochs, avg_val, label=f"{exp_name} (Val, avg)", color=color, linestyle='--')

    # 图表美化
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # # 保存
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plt.savefig(f"loss_curves_{timestamp}.png", dpi=300, bbox_inches='tight')
    # plt.savefig(f"loss_curves_{timestamp}.pdf", bbox_inches='tight')  # 矢量图，适合论文
    # print(f"✅ 图表已保存: loss_curves_{timestamp}.png / .pdf")

    if show_plot:
        plt.show()
    else:
        plt.close()

def check_path(dicts):
    for idx, (exp_name, log_paths) in enumerate(dicts.items()):
        for log_path in log_paths:
            if not os.path.exists(log_path):
                print(f"⚠️ Warning: {log_path} not found, skipping.")
                continue
    

if __name__ == '__main__':
    # experiments_root_path = r'D:\Work\Python\_MSDT\logs_summary'
    # === 配置：log 文件路径与实验分组 ===
    experiments = {
        "Baseline (Adam + WD=0)": [
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_38\Baseline_seed_38.log",   # seed38
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_39\Baseline_seed_39.log",   # seed39
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_40\Baseline_seed_40.log",   # seed40
            r"D:\Work\Python\_MSDT\saved_results\Baseline_seed_48\Baseline_seed_48.log",   # seed48
        ],
        "Adam + WD=1e-5": [
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-05[Seed_48]\Adam_wd_1e-05[Seed_48].log",   # seed48
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-05[Seed_50]\Adam_wd_1e-05[Seed_50].log",   # seed50
        ],
        "Adam + WD=5e-5": [
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_5e-05[Seed_48]\Adam_wd_5e-05[Seed_48].log",   # seed48
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_5e-05[Seed_49]\Adam_wd_5e-05[Seed_49].log",   # seed49
        ],
        "Adam + WD=1e-4": [
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-04[Seed_48]\Adam_wd_1e-04[Seed_48].log",   # seed48
            r"D:\Work\Python\_MSDT\saved_results\Adam_wd_1e-04[Seed_49]\Adam_wd_1e-04[Seed_49].log",   # seed48
        ],
        # "AdamW + WD=0": [
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_0[Seed_48]\AdamW_wd_0[Seed_48].log",   # seed48
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_0[Seed_49]\AdamW_wd_0[Seed_49].log",   # seed49
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_0[Seed_50]\AdamW_wd_0[Seed_50].log",   # seed50
        # ],
        # "AdamW + WD=1e-5": [
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-05[Seed_48]\AdamW_wd_1e-05[Seed_48].log",   # seed48
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-05[Seed_49]\AdamW_wd_1e-05[Seed_49].log",   # seed49
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-05[Seed_50]\AdamW_wd_1e-05[Seed_50].log",   # seed50
        # ],
        # "AdamW + WD=5e-5": [
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_5e-05[Seed_48]\AdamW_wd_5e-05[Seed_48].log",   # seed48
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_5e-05[Seed_49]\AdamW_wd_5e-05[Seed_49].log",   # seed49
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_5e-05[Seed_50]\AdamW_wd_5e-05[Seed_50].log",   # seed50
        # ],
        # "AdamW + WD=1e-4": [
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-04[Seed_48]\AdamW_wd_1e-04[Seed_48].log",   # seed48
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-04[Seed_49]\AdamW_wd_1e-04[Seed_49].log",   # seed49
        #     r"D:\Work\Python\_MSDT\saved_results\AdamW_wd_1e-04[Seed_50]\AdamW_wd_1e-04[Seed_50].log",   # seed50
        # ],
    }
    
    check_path(experiments)
    draw_crave(experiments)

   