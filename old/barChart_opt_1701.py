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

# ================= 2. 数据计算与格式化模块 =================
def format_wd(wd_value):
    """强制统一科学计数法格式，例如 0.0001 -> 1e-4"""
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
        
        # 【修复】更强壮的正则，支持 '0', '1e-05', '0.0001' 等各种写法
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

# ================= 3. 核心绘图引擎 (全新 1x3 极简并排布局) =================
def create_combined_metrics_chart(averaged_data, title="Metrics Comparison", save_path=None):
    models = list(averaged_data.keys())
    if not models: return

    # 指标重新分组：核心主指标、低数值指标(Loss+F1)、高数值指标(ASD)
    metrics_main = ['Dice (DSC)', 'IoU', 'Accuracy', 'PR-AUC', 'Recall', 'Precision']
    metrics_sub  = ['Val Loss', 'Boundary F1 Score']
    metrics_asd  = ['Average Surface Distance (ASD)']

    # 创建 1行3列 画布，赋予极度宽敞的空间
    fig, axes = plt.subplots(3, 1, figsize=(12, 28))
    
    # 全局配色
    colors = ['#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#e7298a', '#a65628', '#f781bf', '#999999']
    n_bars = len(models)
    bar_height = 0.8 / n_bars
    
    # 提取公共绘图逻辑
    def plot_horizontal_group(ax, metrics_list, sub_title):
        y_positions = np.arange(len(metrics_list))
        
        for i, model in enumerate(models):
            vals = [averaged_data[model].get(m, np.nan) for m in metrics_list]
            y_pos = y_positions - 0.4 + (i + 0.5) * bar_height
            
            # 仅在第一个子图绘制图例标签
            bars = ax.barh(y_pos, vals, height=bar_height, label=model if ax == axes[0] else "", 
                           color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)
            
            for bar, m_name in zip(bars, metrics_list):
                w = bar.get_width()
                if not np.isnan(w) and w > 0:
                    # 根据指标类型动态调整格式
                    if 'ASD' in m_name: fmt = '{:.2f}'
                    elif 'Loss' in m_name: fmt = '{:.4f}'
                    else: fmt = '{:.3f}'
                    
                    # 获取该指标的最大值，以计算文字偏移量
                    m_max = max([averaged_data[m].get(m_name, 0) for m in models if not np.isnan(averaged_data[m].get(m_name, 0))] + [0])
                    offset = m_max * 0.02
                    
                    ax.text(w + offset, bar.get_y() + bar.get_height()/2., fmt.format(w),
                            ha='left', va='center', fontsize=13, fontweight='bold')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(metrics_list, fontsize=15, fontweight='bold')
        ax.set_title(sub_title, fontsize=20, fontweight='bold', pad=20)
        ax.grid(True, linestyle='--', alpha=0.6, axis='x')
        
        # 动态设置 X 轴刻度留白 (给文字留出 25% 的空间)
        all_vals = [averaged_data[m].get(met, 0) for m in models for met in metrics_list if not np.isnan(averaged_data[m].get(met, 0))]
        max_val = max(all_vals) if all_vals else 1.0
        ax.set_xlim(0, max_val * 1.25)
        
        # 【排版核心魔法】强制锁定三个子图的 Y 轴范围。
        # 让中间和右侧的小指标紧贴顶部，且柱子粗细与左侧 6 个指标的图完全一模一样！
        ax.set_ylim(len(metrics_main) - 0.5, -0.5)

    # 执行绘图
    plot_horizontal_group(axes[0], metrics_main, "Main Percentage Metrics")
    plot_horizontal_group(axes[1], metrics_sub,  "Loss & Boundary F1 Score")
    plot_horizontal_group(axes[2], metrics_asd,  "Average Surface Distance")

    # 顶置统一图例
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=min(n_bars, 8), 
               fontsize=16, frameon=True, shadow=True)

    # 调整布局间距
    plt.tight_layout(pad=3.0, w_pad=4.0)
    fig.subplots_adjust(top=0.88) # 给图例腾出头部空间
    
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
        print(f"✅ 图表已完美保存至: {save_path}.png")
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
        
        create_combined_metrics_chart(averaged_data, title="Loss Functions Evaluation", save_path=IMG_PATH)
            
    elif PLOT_MODE == 'optimizer':
        adam_res = parse_evaluation_file(os.path.join(BASE_PATH, "Adam.txt"))
        adamw_res = parse_evaluation_file(os.path.join(BASE_PATH, "AdamW.txt"))
        all_results = {**adam_res, **adamw_res}
        
        raw_opt_data = calculate_optimizer_averages(all_results)
        
        flattened_opt_data = {}
        wd_values_sorted = sorted(set(list(raw_opt_data['Adam'].keys()) + list(raw_opt_data['AdamW'].keys())))
        
        for wd in wd_values_sorted:
            # 使用全新的 format_wd 函数，统一生成 0、1e-5、1e-4 的完美后缀
            formatted_wd = format_wd(wd)
            
            if wd in raw_opt_data['Adam']:
                flattened_opt_data[f"Adam (WD={formatted_wd})"] = raw_opt_data['Adam'][wd]
            if wd in raw_opt_data['AdamW']:
                flattened_opt_data[f"AdamW (WD={formatted_wd})"] = raw_opt_data['AdamW'][wd]
                
        create_combined_metrics_chart(flattened_opt_data, title="Optimizer Performance Analysis", save_path=IMG_PATH)