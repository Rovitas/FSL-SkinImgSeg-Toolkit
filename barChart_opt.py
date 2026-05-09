# -*- coding: utf-8 -*-
import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ================= 全局配置 (在这里修改字体大小) =================
VALUE_FONT_SIZE = 16  # 柱子右侧的均值（主数字）大小
STD_FONT_SIZE = 13    # 柱子右侧的标准差（±数）大小
# =================================================================

# ================= 1. 数据解析模块 =================
def parse_evaluation_file(file_path):
    """
    解析评估结果 CSV 文件，提取实验数据
    """
    results = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_name = row.get('Experiment', '').strip()
                if not exp_name:
                    continue
                metrics = {}
                for metric_name, value_str in row.items():
                    if metric_name == 'Experiment':
                        continue          
                    value_str = value_str.strip()
                    if value_str == 'N/A' or not value_str:
                        metrics[metric_name] = np.nan
                    else:
                        try:
                            metrics[metric_name] = float(value_str)
                        except ValueError:
                            metrics[metric_name] = np.nan  
                results[exp_name] = metrics
    except Exception as e:
        print(f"❌ 读取 CSV 文件出错 {file_path}: {e}")
    return results

# ================= 2. 数据计算模块 =================
def format_wd(wd_value):
    if wd_value == 0: return "0"
    return f"{wd_value:.0e}".replace("e-0", "e-")

def calculate_loss_averages(results_dict, config_dict):
    averaged = {}
    for display_name, target_keyword in config_dict.items():
        matched_metrics = [metrics for exp_name, metrics in results_dict.items() if target_keyword in exp_name]
        if matched_metrics:
            metrics_summary = {}
            for metric_name in matched_metrics[0].keys():
                vals = [m[metric_name] for m in matched_metrics if not np.isnan(m[metric_name])]
                if vals:
                    std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    metrics_summary[metric_name] = {'mean': np.mean(vals), 'std': std_val}
                else:
                    metrics_summary[metric_name] = {'mean': np.nan, 'std': np.nan}
            averaged[display_name] = metrics_summary
    return averaged

def calculate_optimizer_averages(results_dict):
    grouped = {'Adam': {}, 'AdamW': {}}
    for exp_name, metrics in results_dict.items():
        opt = 'AdamW' if 'AdamW' in exp_name else 'Adam' if 'Adam' in exp_name else None
        if not opt: continue
        wd_match = re.search(r'wd_([0-9e.-]+)', exp_name)
        if wd_match:
            wd = float(wd_match.group(1))
            grouped[opt].setdefault(wd, []).append(metrics)
            
    averaged = {'Adam': {}, 'AdamW': {}}
    for opt, wd_groups in grouped.items():
        for wd, metrics_list in wd_groups.items():
            metrics_summary = {}
            for metric_name in metrics_list[0].keys():
                vals = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
                if vals:
                    std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    metrics_summary[metric_name] = {'mean': np.mean(vals), 'std': std_val}
                else:
                    metrics_summary[metric_name] = {'mean': np.nan, 'std': np.nan}
            averaged[opt][wd] = metrics_summary
    return averaged

# ================= 3. 表格导出模块 =================
def export_mean_std_table(averaged_data, save_path):
    """生成可以直接复制进 Word/Excel 的 Mean ± STD CSV表格"""
    metrics_order = [
        'Dice (DSC)', 'IoU', 'Accuracy', 'PR-AUC', 'Recall', 'Precision', 
        'Boundary F1 Score', 'Average Surface Distance (ASD)', 'Val Loss'
    ]
    models = list(averaged_data.keys())
    
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Model'] + metrics_order)
        
        for model in models:
            row = [model]
            for m in metrics_order:
                mean_val = averaged_data[model].get(m, {}).get('mean', np.nan)
                std_val  = averaged_data[model].get(m, {}).get('std', 0.0)
                
                if np.isnan(mean_val):
                    row.append("N/A")
                else:
                    if m == 'Average Surface Distance (ASD)':
                        row.append(f"{mean_val:.2f} ± {std_val:.2f}")
                    elif m == 'Val Loss':
                        row.append(f"{mean_val:.4f} ± {std_val:.4f}")
                    else:
                        row.append(f"{mean_val:.3f} ± {std_val:.3f}")
            writer.writerow(row)
    print(f"✅ 学术表格已生成: {save_path}")

# ================= 4. 核心绘图引擎 (生成三个分开的大图) =================
def create_split_reference_charts(averaged_data, base_title="Metrics", save_base_path=None,
                                  value_fontsize=VALUE_FONT_SIZE, std_fontsize=STD_FONT_SIZE):
    models = list(averaged_data.keys())
    if not models: return

    models_reversed = list(reversed(models))
    
    metric_groups = [
        {
            "name": "Part1_Core_Metrics",
            "title": f"{base_title}: Core Segmentation Metrics",
            "metrics": ['Dice (DSC)', 'IoU', 'Accuracy']
        },
        {
            "name": "Part2_Auxiliary_Metrics",
            "title": f"{base_title}: Classification Metrics",
            "metrics": ['PR-AUC', 'Recall', 'Precision']
        },
        {
            "name": "Part3_Distance_and_Loss",
            "title": f"{base_title}: Boundary & Loss Metrics",
            "metrics": ['Boundary F1 Score', 'Average Surface Distance (ASD)', 'Val Loss']
        }
    ]

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0.1, 0.9, len(models_reversed)))

    for group in metric_groups:
        fig, axes = plt.subplots(1, 3, figsize=(24, 10))
        
        for idx, metric in enumerate(group["metrics"]):
            ax = axes[idx]
            
            means = [averaged_data[m].get(metric, {}).get('mean', np.nan) for m in models_reversed]
            stds  = [averaged_data[m].get(metric, {}).get('std', 0.0) for m in models_reversed]
            
            bars = ax.barh(models_reversed, means, height=0.82, color=colors, alpha=0.9,
                           xerr=stds, capsize=6, error_kw={'elinewidth': 1.5, 'ecolor': '#444444'})
            
            ax.set_title(metric, fontsize=20, fontweight='bold', pad=20)
            ax.grid(axis='x', linestyle='--', alpha=0.6, color='gray')
            ax.set_axisbelow(True)
            
            ax.set_yticks(np.arange(len(models_reversed)))
            ax.set_yticklabels(models_reversed, fontsize=15, fontweight='bold')
            ax.tick_params(axis='y', which='major', pad=10)
            ax.tick_params(axis='x', which='major', labelsize=14)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 动态调整X轴范围，确保有充足留白给双行大字
            max_val = max([m + s for m, s in zip(means, stds) if not np.isnan(m)] or [1.0])
            if metric in ['Average Surface Distance (ASD)', 'Val Loss']:
                ax.set_xlim(0, max_val * 1.35) 
            else:
                ax.set_xlim(0, min(1.35, max_val * 1.35)) 
                
            # === 核心修改：均值和标准差差异化标注 ===
            for bar, mean_val, std_val in zip(bars, means, stds):
                w = bar.get_width()
                if not np.isnan(w) and w > 0:
                    fmt = '{:.2f}' if metric in ['Average Surface Distance (ASD)'] else ('{:.4f}' if metric == 'Val Loss' else '{:.3f}')
                    
                    # 确定文字的 X 轴起点：柱子宽 + 标准差长 + 一点点边距
                    text_x = w + std_val + (ax.get_xlim()[1] * 0.015)
                    
                    # 计算 Y 轴上下偏移：使得均值偏上，标准差偏下
                    y_center = bar.get_y() + bar.get_height() / 2.
                    y_mean = y_center + 0.15
                    y_std = y_center - 0.20
                    
                    # 1. 画均值（黑色加粗，字体较大）
                    ax.text(text_x, y_mean, fmt.format(w), ha='left', va='center', 
                            fontsize=value_fontsize, color='black', fontweight='bold')
                    
                    # 2. 画标准差（灰色斜体，字体较小，带括号；并做阈值判断）
                    if std_val > 0:
                        if std_val < 0.001:
                            std_str = "<0.001"
                        else:
                            std_str = fmt.format(std_val)
                            
                        ax.text(text_x, y_std, f"(±{std_str})", ha='left', va='center', 
                                fontsize=std_fontsize, color='#555555', fontstyle='italic')

        fig.suptitle(group["title"], fontsize=26, fontweight='bold', y=1.03)
        plt.tight_layout(pad=2.0, w_pad=3.5)
        
        if save_base_path: 
            save_path = f"{save_base_path}_{group['name']}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 生成子图表: {save_path}")

# ================= 5. 控制入口 =================
if __name__ == "__main__":
    BASE_PATH = r"D:\Work\Python\_MSDT\calcuMetrics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    IMG_BASE_PATH = os.path.join(BASE_PATH, 'image', f'metrics_{timestamp}')
    os.makedirs(os.path.dirname(IMG_BASE_PATH), exist_ok=True)
    
    # [模式切换] : 'loss' 或 'optimizer'
    PLOT_MODE = 'loss' 
    
    if PLOT_MODE == 'loss':
        EXPERIMENTS_TO_PLOT = {
            'BCE(Baseline)': "AdamW_wd_1e-05",
            'Dice':"DiceLoss",
            'Focal':"FocalLoss",
            'Tversky': "TverskyLoss(a0.3_b0.7)",
            'TverskyHD55': "TverskyHD(a0.3_b0.7w0.5_0.5)",
            'TverskyHD82': "TverskyHD(a0.3_b0.7w0.8_0.2)",
            'TverskyHD91': "TverskyHD(a0.3_b0.7w0.9_0.1)"
        }
        file_path = os.path.join(BASE_PATH, "Losses_Comparsion.csv")
        all_results = parse_evaluation_file(file_path)
        averaged_data = calculate_loss_averages(all_results, EXPERIMENTS_TO_PLOT)
        
        # 保留了表格生成功能，但暂时注释掉
        # export_mean_std_table(averaged_data, save_path=f"{IMG_BASE_PATH}_Table.csv")
        create_split_reference_charts(averaged_data, base_title="Loss Ablation", save_base_path=IMG_BASE_PATH)
            
    elif PLOT_MODE == 'optimizer':
        adam_res = parse_evaluation_file(os.path.join(BASE_PATH, "Adam.csv"))
        adamw_res = parse_evaluation_file(os.path.join(BASE_PATH, "AdamW.csv"))
        all_results = {**adam_res, **adamw_res}
        raw_opt_data = calculate_optimizer_averages(all_results)
        flattened_opt_data = {}
        wd_values_sorted = sorted(set(list(raw_opt_data['Adam'].keys()) + list(raw_opt_data['AdamW'].keys())))
        
        for wd in wd_values_sorted:
            formatted_wd = format_wd(wd)
            if wd in raw_opt_data['Adam']: flattened_opt_data[f"Adam (WD={formatted_wd})"] = raw_opt_data['Adam'][wd]
        for wd in wd_values_sorted:
            formatted_wd = format_wd(wd)
            if wd in raw_opt_data['AdamW']: flattened_opt_data[f"AdamW (WD={formatted_wd})"] = raw_opt_data['AdamW'][wd]
                
        # 保留了表格生成功能，但暂时注释掉
        # export_mean_std_table(flattened_opt_data, save_path=f"{IMG_BASE_PATH}_Table.csv")
        create_split_reference_charts(flattened_opt_data, base_title="Optimizer Analysis", save_base_path=IMG_BASE_PATH)