import os
import shutil
from pathlib import Path

def logs_copier(src_dir, dst_dir, max_depth=1):
    src_path = Path(src_dir).resolve()
    
    os.makedirs(dst_dir, exist_ok=True)

    base_depth = len(src_path.parts)

    for root, dirs, files in os.walk(src_path):
        current_depth = len(Path(root).parts) - base_depth
        
        if current_depth > max_depth:
            dirs[:] = []  
            continue

        for file in files:
            if file.endswith('.log'):
                src_file = os.path.join(root, file)
                
                base_name, ext = os.path.splitext(file)
                dst_file = os.path.join(dst_dir, file)
                counter = 1
                
                while os.path.exists(dst_file):
                    dst_file = os.path.join(dst_dir, f"{base_name}_{counter}{ext}")
                    counter += 1
                
                print(f"Copying: {src_file} -> {dst_file}")
                shutil.copy2(src_file, dst_file)


if __name__ == '__main__':
    PROJECT_ROOT = Path( __file__).resolve().parents[2]
    exps_saved_root = os.path.join(PROJECT_ROOT, "saved_results")
    destination_dir = os.path.join(PROJECT_ROOT, "logs_summary")

    logs_copier(exps_saved_root, destination_dir)
