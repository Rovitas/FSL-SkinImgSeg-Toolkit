import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ================= 1. 数据解析模块 =================
def parse_evaluation_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    results = {}
    metric_patterns = {
        'Val Loss': r'Val Loss:\s*([0-9.]+)',
        'Dice (DSC)': r'Dice \(DSC\):\s*([0-9.]+)',
        'IoU': r'IoU:\s*([0-9.]+)',
        'Accuracy': r'Accuracy:\s*([0-9.]+)',
        'Recall': r'Recall:\s*([0-9.]+)',
        'Precision': r'Precision:\s*([0-9.]+)',
        'PR-AUC': r'PR-AUC:\s*([0-9.N/A]+)',
        'Average Surface Distance (ASD)': r'Average Surface Distance \(ASD\):\s*([0-9.]+)',
        'Boundary F1 Score': r'Boundary F1 Score:\s*([0-9.]+)'
    }
    for exp in content.split('------------------------------------------------------------'):
        exp_name_match = re.search(r'Experiment:\s*(.+)', exp)
        if not exp_name_match: continue
        exp_name = exp_name_match.group(1).strip()
        metrics = {}
        for metric, pattern in metric_patterns.items():
            match = re.search(pattern, exp)
            if match:
                val = match.group(1)
                metrics[metric] = np.nan if val == 'N/A' else float(val)
        results[exp_name] = metrics
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
            avg_metrics = {}
            for metric_name in matched_metrics[0].keys():
                vals = [m[metric_name] for m in matched_metrics if not np.isnan(m[metric_name])]
                avg_metrics[metric_name] = np.mean(vals) if vals else np.nan
            averaged[display_name] = avg_metrics
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
            avg_metrics = {}
            for metric_name in metrics_list[0].keys():
                vals = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
                avg_metrics[metric_name] = np.mean(vals) if vals else np.nan
            averaged[opt][wd] = avg_metrics
    return averaged

# ================= 3. 核心绘图引擎 (尺寸升级，撑满版面) =================
def create_split_reference_charts(averaged_data, base_title="Metrics", save_base_path=None):
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
        # 【修改点1】大幅提升高度，20x6 升级为 24x10，让每个子图更加宽大方正
        fig, axes = plt.subplots(1, 3, figsize=(24, 10))
        
        for idx, metric in enumerate(group["metrics"]):
            ax = axes[idx]
            vals = [averaged_data[m].get(metric, np.nan) for m in models_reversed]
            
            # 【修改点2】提升 height(柱子厚度)，0.75 升级为 0.82，减小留白，显得饱满
            bars = ax.barh(models_reversed, vals, height=0.82, color=colors, alpha=0.9)
            
            ax.set_title(metric, fontsize=20, fontweight='bold', pad=20)
            ax.grid(axis='x', linestyle='--', alpha=0.6, color='gray')
            ax.set_axisbelow(True)
            
            # 【修改点3】字号全面适配大图
            ax.set_yticks(np.arange(len(models_reversed)))
            ax.set_yticklabels(models_reversed, fontsize=15, fontweight='bold')
            ax.tick_params(axis='y', which='major', pad=10)
            ax.tick_params(axis='x', which='major', labelsize=14)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if metric in ['Average Surface Distance (ASD)', 'Val Loss']:
                max_val = max([v for v in vals if not np.isnan(v)] or [1.0])
                ax.set_xlim(0, max_val * 1.25) 
            else:
                ax.set_xlim(0, 1.15) # 给右侧数字多留一点点空间
                
            for bar in bars:
                w = bar.get_width()
                if not np.isnan(w) and w > 0:
                    fmt = '{:.2f}' if metric in ['Average Surface Distance (ASD)'] else ('{:.4f}' if metric == 'Val Loss' else '{:.3f}')
                    # 字体也稍微调大到14
                    ax.text(w + (ax.get_xlim()[1] * 0.02), bar.get_y() + bar.get_height()/2., 
                            fmt.format(w), ha='left', va='center', fontsize=14, color='#222222', fontweight='bold')

        fig.suptitle(group["title"], fontsize=26, fontweight='bold', y=1.03)
        plt.tight_layout(pad=2.0, w_pad=3.5)
        
        if save_base_path: 
            os.makedirs(os.path.dirname(save_base_path), exist_ok=True)
            save_path = f"{save_base_path}_{group['name']}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 生成子图表: {save_path}")
        plt.show()

# ================= 4. 控制入口 =================
if __name__ == "__main__":
    BASE_PATH = r"D:\Work\Python\_MSDT\calcuMetrics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    IMG_BASE_PATH = os.path.join(BASE_PATH, 'image', f'metrics_{timestamp}')
    
    # [模式切换] : 'loss' 或 'optimizer'
    PLOT_MODE = 'optimizer' 
    
    if PLOT_MODE == 'loss':
        EXPERIMENTS_TO_PLOT = {
            'Baseline': "Adam_wd_1e-05",
            'TverskyHD55': "TverskyHD(a0.3_b0.7w0.5_0.5)",
            'TverskyHD82': "TverskyHD(a0.3_b0.7w0.8_0.2)",
            'TverskyHD91': "TverskyHD(a0.3_b0.7w0.9_0.1)"
        }
        file_path = os.path.join(BASE_PATH, "Losses_Comparsion.txt")
        all_results = parse_evaluation_file(file_path)
        averaged_data = calculate_loss_averages(all_results, EXPERIMENTS_TO_PLOT)
        create_split_reference_charts(averaged_data, base_title="Loss Ablation", save_base_path=IMG_BASE_PATH)
            
    elif PLOT_MODE == 'optimizer':
        adam_res = parse_evaluation_file(os.path.join(BASE_PATH, "Adam.txt"))
        adamw_res = parse_evaluation_file(os.path.join(BASE_PATH, "AdamW.txt"))
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
                
        create_split_reference_charts(flattened_opt_data, base_title="Optimizer Analysis", save_base_path=IMG_BASE_PATH)