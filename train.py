# -*- coding: utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from losses import TverskyHausdorffLoss

from myDataset import MYDataset, MYDataset_From_Pt
from myModel import UNet

import matplotlib.pyplot as plt

from datetime import datetime
import time

import random
import numpy as np
import yaml # [新增] 用于读取任务队列
from tqdm import tqdm  # [新增] 进度条库

def set_seed(seed=42):
    """设置全局随机种子，确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果用 GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 卷积确定性（牺牲速度）
    torch.backends.cudnn.benchmark = False

def create_unique_dir(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir
    else:
        # 尝试 models_dir_new, models_dir_new1, models_dir_new2, ...
        suffix = "_new"
        counter = 0
        while True:
            if counter == 0:
                new_dir = base_dir + suffix
            else:
                new_dir = f"{base_dir}{suffix}{counter}"
            
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                return new_dir
            counter += 1


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs,
    device,
    base_dir=r'.\\',
    resume_path=None,
    seed=None,
    task_name=None
):
    # ===== 创建本次任务的专属目录 =====
    if task_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir = os.path.join(base_dir, 'saved_results', f'task_{timestamp}')
    else :
        task_dir = os.path.join(base_dir, 'saved_results', task_name)
    
    task_dir = create_unique_dir(task_dir)
    models_dir = os.path.join(task_dir, 'models')
    os.makedirs(models_dir)
    
    log_file = os.path.join(task_dir, f"{task_name}.log")
    plot_path = os.path.join(task_dir, f'seed_{seed}_loss_curve.png')
    # =================================
    
    def log(msg):
        """统一日志输出：控制台 + 文件"""
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
    
    # ===== 基础信息记录 =====
    total_samples = len(train_loader.dataset)
    if total_samples == 0:
        raise ValueError("No training samples found! Check your training dataset path.")
    total_vals = len(val_loader.dataset)
    if total_vals == 0:
        raise ValueError("No validation samples found! Check your validation dataset path.")
    
    log("=" * 60)
    log("🚀 TRAINING STARTED")
    log(f"Task Name: {task_name}")
    if resume_path is not None:
        log(f"Resume model path directory: {os.path.abspath(resume_path)}")
    else : log(f"No resume model is used")
    log(f"Total training samples: {total_samples}")
    log(f"Total validation samples: {total_vals}")
    log(f"Device: {device}")
    log(f"Model: {model.__class__.__name__}")
    log(f"Loss function: {criterion}")
    log(f"Optimizer: {optimizer}")
    log(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    log(f"Batch size: {train_loader.batch_size}")
    log(f"Total epochs: {num_epochs}")
    log(f"Seed sets: {seed}")
    log("=" * 60)
    # =======================
    
    # ===== 加载预训练模型（支持续训）=====
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1
    if resume_path and os.path.exists(resume_path):
        if resume_path is not None:
            log(f"🔄 Loading checkpoint from: {os.path.abspath(resume_path)}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss) 
        best_epoch = checkpoint.get('best_epoch', -1) # [新增]
        log(f"▶ Resuming training from epoch {start_epoch}")
    else:
        log("🆕 Starting training from scratch")
    # ===================================
    
    model = model.to(device)
    train_losses = []
    val_losses = []
    start_time = time.time()
    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            log(f'Epoch {epoch+1}/{num_epochs} started')
            
            model.train()
            running_loss = 0.0
            
            # [修改] 使用 tqdm 包装 train_loader
            train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", leave=False)
            for i, (images, labels) in enumerate(train_pbar):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                
                # [修改] 在进度条右侧实时显示当前 batch 的 Loss
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_train_loss = running_loss / len(train_loader.dataset)

            # 验证过程
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                # [修改] 使用 tqdm 包装 val_loader
                val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val  ", leave=False)
                for images, labels in val_pbar:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_val_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            epoch_duration = time.time() - epoch_start
            train_losses.append(epoch_train_loss)

            log(f" → Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Epoch time: {epoch_duration:.2f}s")
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch + 1
                best_model_path = os.path.join(models_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch, # [新增]
                }, best_model_path)
                log(f"🏆 New best model saved (val_loss={best_val_loss:.6f}) → best_model.pth: {best_model_path}")
            
            # 每10轮或最后一轮保存模型
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(models_dir, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch, # [新增]
                }, checkpoint_path)
                log(f"💾 Saved model: model_epoch_{epoch+1}.pth")
    
    except KeyboardInterrupt:
        log("⚠️ Training interrupted by user (Ctrl+C)")
        raise
    except Exception as e:
        log(f"❌ Training failed with error: {str(e)}")
        raise
    
    finally:
        # 绘制并保存损失曲线
        if train_losses:
            plt.figure(figsize=(10, 6))
            # 动态获取实际跑了多少轮，防止中途报错或人为中断时画图崩溃
            actual_epochs = len(train_losses)
            x_axis = range(start_epoch + 1, start_epoch + actual_epochs + 1)
            plt.plot(x_axis, train_losses, label='Train Loss', linewidth=2)

        # 只有当 val_losses 也有数据时才画它（防止只跑了train没跑val就断了）
            if val_losses and len(val_losses) == actual_epochs:
                plt.plot(x_axis, val_losses, label='Validation Loss', linewidth=2)

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Curve')
            plt.grid(True)
            plt.legend()

            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        # ===== 训练结束总结 =====
        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        log("\n" + "=" * 60)
        log("✅ TRAINING FINISHED")
        log(f"Total epochs completed: {len(train_losses)}")
        log(f"Final loss: {train_losses[-1]:.6f}" if train_losses else "No loss recorded")
        if best_epoch != -1:
            log(f"Best val loss: {best_val_loss:.6f} (achieved at epoch {best_epoch})")
        log(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        log(f"Models saved to: {models_dir}")
        log(f"Loss curve saved: {plot_path}")
        log(f"Full log at: {log_file}")
        log("=" * 60)
        # =========================


def main(run_type="train_with_pt", opt="default", crit="default", set_decay=1e-5,
        loss_params = {
            'focal': {'alpha': 0.25, 'gamma': 2},
            'tversky': {'alpha': 0.3, 'beta': 0.7},
            'tversky_hd': {'alpha': 0.3, 'beta': 0.7, 'w_tv': 0.8, 'w_hd': 0.2}
        },
        resume_path=None, seed=48, epochs=50, use_aug=False):
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = r"d:\Work\Python\_MSDT"
    TASK_NAME = None
    EPOCHS = epochs
    model = UNet().to(device)
    #   Adam Vs AdamW
    if opt.lower() in ("adamw", "default"):
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=set_decay)
    elif opt.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=set_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=set_decay)
        print(f"⚠️ Unknown type of optimizer: {opt}, using AdamW as default.")
    if opt.lower() != "default":
        if set_decay == 0:
            TASK_NAME = f"{str(optimizer).split(' (')[0]}_wd_{set_decay}[Seed_{seed}]"
        else:
            TASK_NAME = f"{str(optimizer).split(' (')[0]}_wd_{set_decay:.0e}[Seed_{seed}]"

    #   Loss Functions
    if crit.lower() in ("bce", "default"):
        criterion = nn.BCEWithLogitsLoss()
        TASK_NAME = f"BCELoss[Seed_{seed}]"
    elif crit.lower() == "dice":
        criterion = smp.losses.DiceLoss(mode="binary")
        TASK_NAME = f"{str(criterion).split('(')[0]}[Seed_{seed}]"
    elif crit.lower() == "focal":
        # criterion = smp.losses.FocalLoss(mode="binary", gamma=2)    # 默认
        # criterion = smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=1)
        # criterion = smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2)    # 默认
        criterion = smp.losses.FocalLoss(mode="binary", alpha=loss_params['focal']['alpha'], gamma=loss_params['focal']['gamma'])
        TASK_NAME = f"FocalLoss(a{loss_params['focal']['alpha']}_g{loss_params['focal']['gamma']})[Seed_{seed}]"
    elif crit.lower() == "tversky":
        # criterion = smp.losses.TverskyLoss(mode="binary", alpha=0.5, beta=0.5)  # 默认（等价 Dice）
        # criterion = smp.losses.TverskyLoss(mode="binary", alpha=0.3, beta=0.7)  # 重视召回（减少 FN)
        # criterion = smp.losses.TverskyLoss(mode="binary", alpha=0.2, beta=0.8)  # 极端重视召回
        criterion = smp.losses.TverskyLoss(mode="binary", alpha=loss_params['tversky']['alpha'], beta=loss_params['tversky']['beta'])
        TASK_NAME = f"TverskyLoss(a{loss_params['tversky']['alpha']}_b{loss_params['tversky']['beta']})[Seed_{seed}]"
    elif crit.lower() == "tversky_hd":
        criterion = TverskyHausdorffLoss(
            tversky_alpha=loss_params['tversky_hd']['alpha'],
            tversky_beta=loss_params['tversky_hd']['beta'],
            hd_alpha=1.0,
            hd_sample_points=100,
            weight_tversky=0.8,
            weight_hd=0.2
        )
        TASK_NAME = f"TverskyHD(a{loss_params['tversky_hd']['alpha']}_b{loss_params['tversky_hd']['beta']}w{loss_params['tversky_hd']['w_tv']}_{loss_params['tversky_hd']['w_hd']})[Seed_{seed}]"
    elif crit.lower() == "bcedice":
        criterion = lambda pred, target: (
            0.5 * smp.losses.DiceLoss(mode="binary")(pred, target) +
            0.5 * nn.BCEWithLogitsLoss()(pred, target)
        )
        TASK_NAME = f"BCEDiceLoss[Seed_{seed}]"
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"⚠️ Unknown criterion: {crit}, using BCEWithLogitsLoss as default.")   

    if resume_path is not None:
        TASK_NAME = f"(Resume_from_ep{os.path.basename(resume_path).split('.')[0].split('_')[-1]})" + TASK_NAME

    # Run Type (Run with which dataset)
    if run_type.lower() == "train_with_img":
        TRAIN_DIR = r"d:\Work\Python\_MSDT\images_split\train"
        VAL_DIR = r"d:\Work\Python\_MSDT\images_split\val"
    elif run_type.lower() == "debug":
        # 调试代码用的极少量数据
        TRAIN_DIR = r"d:\Work\Python\_MSDT\imagesT"
        VAL_DIR = r"d:\Work\Python\_MSDT\imagesT"
        TASK_NAME = f"(Debug){TASK_NAME}"
        EPOCHS = 3 
    elif run_type.lower() == "train_with_pt":
        if use_aug:
            PT_TRAIN = r"d:\Work\Python\_MSDT\dataset\train_aug.pt"
            print("Using augmentation, loading 'train_aug.pt'")
        else:
            PT_TRAIN = r"d:\Work\Python\_MSDT\dataset\train.pt"
            print("Using original, loading 'train.pt'")
        PT_VAL = r"d:\Work\Python\_MSDT\dataset\val.pt"
        transform = transforms.Compose([transforms.Resize((256, 256))])
        train_dataset = MYDataset_From_Pt(pt_file=PT_TRAIN, transform=transform)
        val_dataset = MYDataset_From_Pt(pt_file=PT_VAL, transform=transform)
    else :
        print(f"Training failed with error: Wrong 'run_type':{run_type}")

    if run_type.lower() == "train_with_img" or run_type.lower() == "debug" :
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        train_dataset = MYDataset(base_dir=TRAIN_DIR, transform=transform)
        val_dataset = MYDataset(base_dir=VAL_DIR, transform=transform)

    if use_aug:
        TASK_NAME += "_Aug"

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        device=device,
        base_dir=BASE_DIR,
        resume_path=resume_path,
        seed=seed,
        task_name=TASK_NAME
    )

# ================== [新增] 动态任务队列管理器 ==================
def get_next_task(queue_file="tasks.yml", history_file="tasks_history.yml"):
    """读取并弹出队列中的第一个任务，同时自动备份到历史档案中"""
    if not os.path.exists(queue_file):
        return None
    
    try:
        with open(queue_file, 'r', encoding='utf-8') as f:
            tasks = yaml.safe_load(f)
            
        if not tasks or len(tasks) == 0:
            return None
            
        # 1. 弹出第一个任务
        current_task = tasks.pop(0)
        
        # 2. 更新待办列表 (覆盖写入 tasks.yml)
        with open(queue_file, 'w', encoding='utf-8') as f:
            yaml.dump(tasks, f, allow_unicode=True, default_flow_style=False)
            
        # 3. 自动备份到历史记录档案
        # 给任务盖个时间戳印章，方便以后查阅
        archive_task = current_task.copy()
        archive_task["_executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as hf:
                    loaded_history = yaml.safe_load(hf)
                    if loaded_history is not None:  # 🟢 确保文件不为空
                        history = loaded_history
            except yaml.YAMLError as inner_exc:
                print(f"详细报错信息: \n{inner_exc}")
                # 如果历史文件为空或损坏，新建一个列表
                
        history.append(archive_task)
        
        # 写入历史文件
        with open(history_file, 'w', encoding='utf-8') as hf:
            yaml.dump(history, hf, allow_unicode=True, default_flow_style=False)
            
        return current_task
        
    except yaml.YAMLError as exc:
        print(f"⚠️ {queue_file} 格式错误，请检查 YAML 格式！程序将暂停 10 秒后重试...")
        print(f"详细报错信息: \n{exc}")
        time.sleep(10)
        return get_next_task(queue_file, history_file)
    except Exception as e:
        print(f"⚠️ 读取任务队列失败: \n{e}")
        return None


if __name__ == '__main__':
    QUEUE_FILE = r"d:\Work\Python\_MSDT\logs\tasks.yml"
    HISTORY_FILE = r"d:\Work\Python\_MSDT\logs\tasks_history.yml"

    # 如果没有找到 tasks.yml，自动生成一个模板文件
    if not os.path.exists(QUEUE_FILE):
        tasks_yml_template = """# ==========================================
# 实验任务配置清单 (YAML格式支持直接写注释)
# ==========================================
# 备忘录 / 参数说明：
# run_type    : 不填默认"train_with_pt"，可选 "train_with_pt", "train_with_img" (正式跑配置的epochs) 或 "debug" (快速测试3轮, 忽略外部传参的epochs)
# crit        : 损失函数，不填默认"bce"，可选 "bce", "dice", "focal", "tversky", "tversky_hd"
# use_aug     : 不填默认"False"，True (开启数据增强) 或 False (关闭)
# resume_path : 断点续训模型路径 (绝对路径)，如果没有可以不填
# ==========================================

# 示例 1: 这是一个用于调试的短任务
- run_type: debug
  crit: dice
  seed: 48
  use_aug: False
  epochs: 3

# 示例 2: 这是一个正式跑 50 轮的 tversky_hd 实验
- run_type: train_with_pt
  crit: tversky_hd
  seed: 48
  epochs: 50
  loss_params: {'tversky_hd': {'alpha': 0.3, 'beta': 0.7, 'w_tv': 0.8, 'w_hd': 0.2}}
"""

        with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
            f.write(tasks_yml_template)
        print(f"📝 未检测到 {QUEUE_FILE}，已自动生成模板文件！")
        print("请打开 tasks.yml 修改为你想要的任务，然后重新运行代码。")
        exit(0)

    print(f"📂 任务监控已启动。执行历史将自动归档至: {HISTORY_FILE}")

    # 循环监听并执行任务队列
    while True:
        task_kwargs = get_next_task(QUEUE_FILE, HISTORY_FILE)
        
        if task_kwargs is None:
            print("\n✅ 所有任务已执行完毕！任务队列 (tasks.yml) 为空。")
            break
            
        print("\n" + "="*50)
        print("▶️ 加载到新任务，参数如下：")
        for k, v in task_kwargs.items():
            print(f"   {k}: {v}")
        print("="*50 + "\n")
        
        # 执行主训练任务
        main(**task_kwargs)
        
        # 任务之间稍微暂停 2 秒，缓冲一下释放显存
        time.sleep(2)