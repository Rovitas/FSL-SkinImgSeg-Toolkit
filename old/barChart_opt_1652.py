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
        wd_match = re.search(r'wd_(\d+e?-?\d+)', exp_name)
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

# 格式化科学计数法的辅助函数 (例如: 把 0.0001 转成 1e-4)
def format_wd(wd_value):
    if wd_value == 0: return "0"
    # :.0e 会生成 1e-04，replace 将其转换为你想要的 1e-4
    return f"{wd_value:.0e}".replace("e-0", "e-")

# ================= 3. 核心绘图引擎 (全新左右布局) =================
def create_combined_metrics_chart(averaged_data, title="Metrics Comparison", save_path=None):
    models = list(averaged_data.keys())
    if not models: return

    # 指标分类：百分比类型(左边) vs 绝对值类型(右边)
    metrics_main = ['Dice (DSC)', 'IoU', 'Accuracy', 'PR-AUC', 'Recall', 'Precision', 'Boundary F1 Score']
    metrics_loss = 'Val Loss'
    metrics_asd = 'Average Surface Distance (ASD)'

    # 构建布局：1行2列。左边列画大图，右边列切分为上下2行
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], height_ratios=[1, 1], wspace=0.15, hspace=0.35)
    
    ax_main = fig.add_subplot(gs[:, 0])   # 占据左侧整列
    ax_loss = fig.add_subplot(gs[0, 1])   # 右上
    ax_asd = fig.add_subplot(gs[1, 1])    # 右下

    # 颜色库
    colors = ['#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#e7298a', '#a65628', '#f781bf', '#999999']
    n_bars = len(models)
    
    # -----------------------
    # 绘制左侧 (水平条形图 - 保证数字正立)
    # -----------------------
    n_groups = len(metrics_main)
    bar_height = 0.8 / n_bars
    y_main = np.arange(n_groups)

    for i, model in enumerate(models):
        vals = [averaged_data[model].get(m, np.nan) for m in metrics_main]
        # 计算每个柱子的垂直位置
        y_pos = y_main - 0.4 + (i + 0.5) * bar_height
        
        bars = ax_main.barh(y_pos, vals, height=bar_height, label=model, 
                            color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)
        
        # 添加正立的数值标签（放在柱子右侧）
        for bar in bars:
            w = bar.get_width()
            if not np.isnan(w) and w > 0:
                ax_main.text(w + 0.015, bar.get_y() + bar.get_height()/2., f'{w:.3f}',
                             ha='left', va='center', fontsize=12, fontweight='bold')

    ax_main.set_yticks(y_main)
    ax_main.set_yticklabels(metrics_main, fontsize=15, fontweight='bold')
    ax_main.set_xlim(0, 1.15)  # X轴留点空间给文字
    ax_main.invert_yaxis()     # 翻转Y轴，让第一个指标在最上面
    ax_main.grid(True, linestyle='--', alpha=0.6, axis='x')
    
    # 将图例放在左侧图表的顶部中心
    ax_main.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=min(n_bars, 4), fontsize=14, frameon=True)
    ax_main.set_title(f"{title} (Percentage Metrics)", pad=70, fontsize=22, fontweight='bold')

    # -----------------------
    # 绘制右上角 (Val Loss)
    # -----------------------
    x_other = np.arange(n_bars)
    vals_loss = [averaged_data[m].get(metrics_loss, np.nan) for m in models]
    
    bars_loss = ax_loss.bar(x_other, vals_loss, color=[colors[i%len(colors)] for i in range(n_bars)], width=0.55)
    ax_loss.set_xticks(x_other)
    # 文字太长时，微调倾斜角度，防止与横坐标轴重叠
    ax_loss.set_xticklabels(models, fontsize=12, rotation=15, ha='right')
    ax_loss.set_title(metrics_loss, fontsize=18, fontweight='bold')
    ax_loss.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    max_loss = max([v for v in vals_loss if not np.isnan(v)] or [1.0])
    ax_loss.set_ylim(0, max_loss * 1.25)
    for bar in bars_loss:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax_loss.text(bar.get_x() + bar.get_width()/2., h + max_loss*0.02, f'{h:.4f}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

    # -----------------------
    # 绘制右下角 (ASD)
    # -----------------------
    vals_asd = [averaged_data[m].get(metrics_asd, np.nan) for m in models]
    
    bars_asd = ax_asd.bar(x_other, vals_asd, color=[colors[i%len(colors)] for i in range(n_bars)], width=0.55)
    ax_asd.set_xticks(x_other)
    ax_asd.set_xticklabels(models, fontsize=12, rotation=15, ha='right')
    ax_asd.set_title(metrics_asd, fontsize=18, fontweight='bold')
    ax_asd.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    max_asd = max([v for v in vals_asd if not np.isnan(v)] or [10.0])
    ax_asd.set_ylim(0, max_asd * 1.25)
    for bar in bars_asd:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax_asd.text(bar.get_x() + bar.get_width()/2., h + max_asd*0.02, f'{h:.2f}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

    # -----------------------
    # 保存与显示
    # -----------------------
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存至: {save_path}.png")
    plt.show()

# ================= 4. 控制入口 =================
if __name__ == "__main__":
    BASE_PATH = r"D:\Work\Python\_MSDT\calcuMetrics"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    IMG_PATH = os.path.join(BASE_PATH, 'image', f'comparison_metrics_{timestamp}')
    
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
        
        create_combined_metrics_chart(
            averaged_data, 
            title="Loss Functions Evaluation",
            save_path=IMG_PATH
        )
            
    elif PLOT_MODE == 'optimizer':
        adam_res = parse_evaluation_file(os.path.join(BASE_PATH, "Adam.txt"))
        adamw_res = parse_evaluation_file(os.path.join(BASE_PATH, "AdamW.txt"))
        all_results = {**adam_res, **adamw_res}
        
        raw_opt_data = calculate_optimizer_averages(all_results)
        
        # 将嵌套的字典扁平化，生成统一标准格式如 "AdamW (WD=1e-4)"
        flattened_opt_data = {}
        wd_values_sorted = sorted(set(list(raw_opt_data['Adam'].keys()) + list(raw_opt_data['AdamW'].keys())))
        
        for wd in wd_values_sorted:
            # 核心修改：统一调用 format_wd 方法进行格式化
            formatted_wd = format_wd(wd)
            
            if wd in raw_opt_data['Adam']:
                flattened_opt_data[f"Adam (WD={formatted_wd})"] = raw_opt_data['Adam'][wd]
            if wd in raw_opt_data['AdamW']:
                flattened_opt_data[f"AdamW (WD={formatted_wd})"] = raw_opt_data['AdamW'][wd]
                
        create_combined_metrics_chart(
            flattened_opt_data,
            title="Optimizer Performance Analysis",
            save_path=IMG_PATH
        )