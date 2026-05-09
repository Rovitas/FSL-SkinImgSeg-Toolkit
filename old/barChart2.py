import re
import matplotlib.pyplot as plt
import numpy as np

def parse_evaluation_file(file_path):
    """
    解析评估结果txt文件，提取实验数据
    
    Args:
        file_path: txt文件路径
        
    Returns:
        dict: {experiment_name: {metric: value}}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按实验分割
    experiments = content.split('------------------------------------------------------------')
    
    results = {}
    
    for exp in experiments:
        if 'Experiment:' not in exp:
            continue
            
        # 提取实验名称
        exp_name_match = re.search(r'Experiment:\s*(.+)', exp)
        if not exp_name_match:
            continue
        exp_name = exp_name_match.group(1).strip()
        
        # 提取所有指标
        metrics = {}
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
        
        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, exp)
            if match:
                value_str = match.group(1)
                if value_str == 'N/A':
                    metrics[metric_name] = np.nan
                else:
                    metrics[metric_name] = float(value_str)
        
        results[exp_name] = metrics
    
    return results

# 定义损失函数映射
LOSS_FUNCTIONS = {
    'BCE': "AdamW_wd_1e-05",
    'Dice': "DiceLoss",
    'Focal': "FocalLoss(a0.25_g2)",
    'Tversky': "TverskyLoss(a0.3_b0.7)",
    'Combo': "TverskyHD(a0.3_b0.7w0.8_0.2)"
}

def extract_loss_function(experiment_name):
    """
    从实验名称中提取损失函数类型
    根据您提供的 LOSS_FUNCTIONS 映射进行匹配
    """
    loss_functions = ['BCE', 'Dice', 'Focal', 'Tversky', 'Combo']
    
    for loss_type in loss_functions:
        # 检查实验名称是否包含对应损失函数的标识符
        if LOSS_FUNCTIONS[loss_type] in experiment_name:
            return loss_type
    
    # 如果没找到，返回默认值
    return "Unknown"

def group_experiments_by_loss_function(results_dict):
    """
    将实验按损失函数类型分组
    """
    grouped = {}
    
    for exp_name, metrics in results_dict.items():
        loss_func = extract_loss_function(exp_name)
        if loss_func not in grouped:
            grouped[loss_func] = []
        grouped[loss_func].append(metrics)
    
    return grouped

def calculate_average_metrics(grouped_data):
    """
    计算每个损失函数的平均指标
    
    Args:
        grouped_data: 分组后的数据
        
    Returns:
        dict: {loss_function: avg_metrics}
    """
    averaged = {}
    
    for loss_func, metrics_list in grouped_data.items():
        if not metrics_list:
            continue
            
        # 计算每个指标的平均值
        avg_metrics = {}
        metric_keys = metrics_list[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in metrics_list if not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = np.nan
                
        averaged[loss_func] = avg_metrics
    
    return averaged

def create_spacious_loss_comparison_chart(averaged_data, save_path=None, figsize=(18, 12)):
    """
    创建宽敞版多指标对比图（优化布局，减少拥挤）
    """
    # 确保按照指定顺序显示损失函数
    loss_functions = ['BCE', 'Dice', 'Focal', 'Tversky', 'Combo']
    
    # 定义要绘制的指标
    metrics_to_plot = [
        'Dice (DSC)',
        'IoU',
        'Accuracy',
        'PR-AUC',
        'Val Loss',
        'Recall',
        'Precision',
        'Average Surface Distance (ASD)',
        'Boundary F1 Score'
    ]
    
    # 创建3x3的子图布局，但大幅增加间距和边距
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    # fig.suptitle('Comparison of 5 Loss Functions Across Multiple Metrics', 
    #              fontsize=20, fontweight='bold', y=0.98)
    
    # 颜色方案 - 与您提供的图片一致
    colors = ['#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#e7298a']  # 橙、蓝、绿、紫、粉
    
    # 为每个指标创建子图
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # 获取该指标的值
        metric_values = []
        for loss_func in loss_functions:
            if loss_func in averaged_data and metric in averaged_data[loss_func]:
                metric_values.append(averaged_data[loss_func][metric])
            else:
                metric_values.append(np.nan)
        
        # 创建柱状图
        x = np.arange(len(loss_functions))
        bars = ax.bar(x, metric_values, color=colors[:len(loss_functions)], width=0.6)
        
        # 设置标题和标签
        ax.set_title(metric, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(['BCE', 'Dice', 'Focal', 'Tversky', 'Combo'], fontsize=11)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                # 对于不同类型的指标调整标签位置和格式
                if metric == 'Average Surface Distance (ASD)' or metric == 'Val Loss':
                    # 大数值指标：显示2位小数，放在柱子上方
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.03,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                elif metric == 'Boundary F1 Score':
                    # 小数值指标：显示3位小数
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                else:
                    # 标准指标：显示3位小数
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 设置y轴范围和刻度
        if metric == 'Average Surface Distance (ASD)':
            max_val = max([v for v in metric_values if not np.isnan(v)]) if any(not np.isnan(v) for v in metric_values) else 10
            ax.set_ylim(0, max_val * 1.15)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        elif metric == 'Val Loss':
            max_val = max([v for v in metric_values if not np.isnan(v)]) if any(not np.isnan(v) for v in metric_values) else 1
            ax.set_ylim(0, max_val * 1.15)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        else:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # 增加轴标签字体大小
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 移除多余的子图（如果指标少于9个）
    if len(metrics_to_plot) < 9:
        for idx in range(len(metrics_to_plot), 9):
            row = idx // 3
            col = idx % 3
            axes[row, col].set_visible(False)
    
    # 添加统一的图例（放在图表右侧）
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], label=loss_functions[i]) 
                      for i in range(len(loss_functions))]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.02, 0.5), 
               title='Loss Functions', fontsize=12, title_fontsize=13, frameon=True)
    
    # 调整布局参数，大幅增加间距
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)
    
    # 额外增加顶部和右侧空间以容纳图例
    fig.subplots_adjust(right=0.85, top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()

def main():
    """
    主函数：读取文件、处理数据、生成图表
    """
    # 文件路径（请根据实际情况修改）
    evaluation_file = r"D:\Work\Python\_MSDT\calcuMetrics\Losses_Comparsion.txt"
    
    try:
        # 解析评估结果文件
        print("Parsing evaluation results...")
        all_results = parse_evaluation_file(evaluation_file)
        print(f"Found {len(all_results)} experiments")
        
        # 分组和平均
        grouped_data = group_experiments_by_loss_function(all_results)
        averaged_data = calculate_average_metrics(grouped_data)
        
        # 打印平均结果用于验证
        print("\nAverage Results:")
        print("=" * 60)
        for loss_func in ['BCE', 'Dice', 'Focal', 'Tversky', 'Combo']:
            if loss_func in averaged_data:
                dsc = averaged_data[loss_func].get('Dice (DSC)', 'N/A')
                print(f"  {loss_func}: DSC={dsc:.4f}")
            else:
                print(f"  {loss_func}: No data found")
        
        # 创建宽敞版对比图
        create_spacious_loss_comparison_chart(
            averaged_data, 
            save_path="loss_comparison_all_metrics_spacious.png"
        )
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please make sure the input file exists and update the file path in the code.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()