import os
import shutil
from pathlib import Path

def logsCopier(src_dir, dst_dir, max_depth=1):
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # 计算源目录的初始深度（用于比较）
    base_depth = len(src_path.parts)

    for root, dirs, files in os.walk(src_path):
        # 当前目录深度 = 当前路径部分数 - 基础路径部分数
        current_depth = len(Path(root).parts) - base_depth
        
        # 如果超过最大深度，跳过该目录及其子目录
        if current_depth > max_depth:
            dirs[:] = []  # 清空 dirs 防止继续递归（重要！）
            continue

        for file in files:
            if file.endswith('.log'):
                src_file = os.path.join(root, file)
                dst_file = dst_path / file
                
                # 如果目标已存在，可以选择跳过或覆盖
                # 这里默认覆盖（shutil.copy 会覆盖）
                print(f"Copying: {src_file} -> {dst_file}")
                shutil.copy2(src_file, dst_file)  # copy2 保留元数据


if __name__ == '__main__':
    exps_saved_root = r"d:\Work\Python\_MSDT\saved_results"
    destination_dir = r"d:\Work\Python\_MSDT\logs_summary"
    logsCopier(exps_saved_root, destination_dir)
