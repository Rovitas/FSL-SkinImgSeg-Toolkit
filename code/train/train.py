# -*- coding: utf-8 -*-
import os
import random
import time
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import TverskyHausdorffLoss
from my_dataset import MYDataset, MYDataset_From_Pt
from my_model import UNet

PROJECT_ROOT = Path( __file__).resolve().parents[2]

def set_seed(seed=42):
    """全局随机种子设置"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_unique_dir(base_dir):
    """创建唯一目录，防止覆盖"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir
    
    suffix = "_new"
    counter = 0
    while True:
        # 如果 counter 为 0，不加数字；否则加上后缀数字
        new_dir = f"{base_dir}{suffix}{counter if counter > 0 else ''}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        counter += 1

def setup_logger(log_file):
    """
    配置 logging 模块，取代原始文件 I/O，实现控制台和文件的双轨高效写入。
    """
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_dataloaders(run_type, use_aug, project_root):
    """统一的数据集构建分支逻辑"""
    run_type = run_type.lower()
    
    if run_type == "debug":
        # 直接复用 train_with_pt 的路径逻辑
        pt_filename = "train_aug.pt" if use_aug else "train.pt"
        pt_train = os.path.join(project_root, "pt_dataset", pt_filename)
        pt_val = os.path.join(project_root, "pt_dataset", "val.pt")
        transform = transforms.Compose([transforms.Resize((256, 256))])
        
        logging.info(f"[Debug Mode] Loading PT dataset and slicing top 5 samples.")
        full_train_dataset = MYDataset_From_Pt(pt_file=pt_train, transform=transform)
        full_val_dataset = MYDataset_From_Pt(pt_file=pt_val, transform=transform)
        
        train_dataset = torch.utils.data.Subset(full_train_dataset, range(min(5, len(full_train_dataset))))
        val_dataset = torch.utils.data.Subset(full_val_dataset, range(min(5, len(full_val_dataset))))
        
    elif run_type == "train_with_img":
        train_dir = os.path.join(project_root, "images_split", "train")
        val_dir = os.path.join(project_root, "images_split", "val")
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        train_dataset = MYDataset(base_dir=train_dir, transform=transform)
        val_dataset = MYDataset(base_dir=val_dir, transform=transform)
        
    elif run_type == "train_with_pt":
        pt_filename = "train_aug.pt" if use_aug else "train.pt"
        pt_train = os.path.join(project_root, "pt_dataset", pt_filename)
        pt_val = os.path.join(project_root, "pt_dataset", "val.pt")
        transform = transforms.Compose([transforms.Resize((256, 256))])
        
        logging.info(f"Loading PT dataset: {pt_filename}")
        train_dataset = MYDataset_From_Pt(pt_file=pt_train, transform=transform)
        val_dataset = MYDataset_From_Pt(pt_file=pt_val, transform=transform)
    else:
        raise ValueError(f"Unknown run_type: {run_type}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    return train_loader, val_loader

def get_optimizer(opt_name, model, set_decay, seed):
    """解析并构造优化器"""
    opt_name = opt_name.lower()
    if opt_name in ("adamw", "default"):
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=set_decay)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=set_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=set_decay)
        logging.warning(f"⚠️ Unknown optimizer: {opt_name}, using AdamW as default.")
        
    # 构建基础后缀名称
    opt_str = str(optimizer).split(' (')[0]
    if opt_name != "default":
        suffix = f"_{opt_str}_wd_{set_decay}[Seed_{seed}]" if set_decay == 0 else f"_{opt_str}_wd_{set_decay:.0e}[Seed_{seed}]"
    else:
        suffix = ""
    return optimizer, suffix

def get_criterion(crit_name, loss_params, seed):
    """解析并构造损失函数"""
    crit_name = crit_name.lower()
    
    if crit_name in ("bce", "default"):
        criterion = nn.BCEWithLogitsLoss()
        task_name = f"BCELoss[Seed_{seed}]"
    elif crit_name == "dice":
        criterion = smp.losses.DiceLoss(mode="binary")
        task_name = f"{str(criterion).split('(')[0]}[Seed_{seed}]"
    elif crit_name == "focal":
        p = loss_params['focal']
        criterion = smp.losses.FocalLoss(mode="binary", alpha=p['alpha'], gamma=p['gamma'])
        task_name = f"FocalLoss(a{p['alpha']}_g{p['gamma']})[Seed_{seed}]"
    elif crit_name == "tversky":
        p = loss_params['tversky']
        criterion = smp.losses.TverskyLoss(mode="binary", alpha=p['alpha'], beta=p['beta'])
        task_name = f"TverskyLoss(a{p['alpha']}_b{p['beta']})[Seed_{seed}]"
    elif crit_name == "tversky_hd":
        p = loss_params['tversky_hd']
        criterion = TverskyHausdorffLoss(
            tversky_alpha=p['alpha'], tversky_beta=p['beta'], hd_alpha=1.0, hd_sample_points=100,
            weight_tversky=p['w_tv'], weight_hd=p['w_hd']
        )
        task_name = f"TverskyHD(a{p['alpha']}_b{p['beta']}w{p['w_tv']}_{p['w_hd']})[Seed_{seed}]"
    elif crit_name == "bcedice":
        criterion = lambda pred, target: (0.5 * smp.losses.DiceLoss(mode="binary")(pred, target) + 0.5 * nn.BCEWithLogitsLoss()(pred, target))
        task_name = f"BCEDiceLoss[Seed_{seed}]"
    else:
        criterion = nn.BCEWithLogitsLoss()
        task_name = f"BCELoss[Seed_{seed}]"
        logging.warning(f"⚠️ Unknown criterion: {crit_name}, using BCE as default.")
        
    return criterion, task_name


def train_model(
    model, criterion, optimizer, train_loader, val_loader, 
    num_epochs, device, task_dir, resume_path=None, seed=None, task_name=None
):
    models_dir = os.path.join(task_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    log_file = os.path.join(task_dir, f"{task_name}.log")
    plot_path = os.path.join(task_dir, f'seed_{seed}_loss_curve.png')
    
    logger = setup_logger(log_file)
    
    total_samples = len(train_loader.dataset)
    total_vals = len(val_loader.dataset)
    if total_samples == 0 or total_vals == 0:
        raise ValueError("No training or validation samples found! Check your dataset path.")
    
    logger.info("=" * 60)
    logger.info("🚀 TRAINING STARTED")
    logger.info(f"Task Name: {task_name}")
    logger.info(f"Resume path: {os.path.abspath(resume_path) if resume_path else 'None'}")
    logger.info(f"Total training samples: {total_samples}")
    logger.info(f"Total validation samples: {total_vals}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Loss function: {criterion}")
    logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"Seed sets: {seed}")
    logger.info("=" * 60)
    
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1
    
    if resume_path and os.path.exists(resume_path):
        logger.info(f"🔄 Loading checkpoint from: {os.path.abspath(resume_path)}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss) 
        best_epoch = checkpoint.get('best_epoch', -1)
        logger.info(f"▶ Resuming training from epoch {start_epoch}")
    else:
        logger.info("🆕 Starting training from scratch")
    
    model = model.to(device)
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            logger.info(f'Epoch {epoch+1}/{num_epochs} started')
            
            model.train()
            running_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", leave=False)
            for images, labels in train_pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val  ", leave=False)
                for images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_val_loss = val_running_loss / len(val_loader.dataset)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            epoch_duration = time.time() - epoch_start
            logger.info(f" → Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Epoch time: {epoch_duration:.2f}s")
            
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
                    'best_epoch': best_epoch,
                }, best_model_path)
                logger.info(f"🏆 New best model saved (val_loss={best_val_loss:.6f}) → best_model.pth")
            
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(models_dir, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                }, checkpoint_path)
                logger.info(f"💾 Saved model: model_epoch_{epoch+1}.pth")
    
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        raise
    
    finally:
        # [优化] 增加异常处理和中断期间绘图崩溃的防护校验
        actual_epochs = len(train_losses)
        if actual_epochs > 0 and len(train_losses) == len(val_losses):
            plt.figure(figsize=(10, 6))
            x_axis = range(start_epoch + 1, start_epoch + actual_epochs + 1)
            plt.plot(x_axis, train_losses, label='Train Loss', linewidth=2)
            plt.plot(x_axis, val_losses, label='Validation Loss', linewidth=2)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Curve')
            plt.grid(True)
            plt.legend()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING FINISHED")
        logger.info(f"Total epochs completed: {actual_epochs}")
        logger.info(f"Final loss: {train_losses[-1]:.6f}" if train_losses else "No loss recorded")
        if best_epoch != -1:
            logger.info(f"Best val loss: {best_val_loss:.6f} (achieved at epoch {best_epoch})")
        logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        logger.info(f"Models saved to: {models_dir}")
        logger.info(f"Loss curve saved: {plot_path}")
        logger.info("=" * 60)


def main(run_type="train_with_pt", opt="default", crit="default", set_decay=1e-5,
        loss_params=None, resume_path=None, seed=48, epochs=50, use_aug=False):
    
    if loss_params is None:
        loss_params = {
            'focal': {'alpha': 0.25, 'gamma': 2},
            'tversky': {'alpha': 0.3, 'beta': 0.7},
            'tversky_hd': {'alpha': 0.3, 'beta': 0.7, 'w_tv': 0.8, 'w_hd': 0.2}
        }
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # 通过解耦后的工厂函数配置模型
    criterion, base_task_name = get_criterion(crit, loss_params, seed)
    optimizer, opt_suffix = get_optimizer(opt, model, set_decay, seed)
    
    # 动态组装任务名称
    task_name = base_task_name + opt_suffix
    if resume_path is not None:
        ep = os.path.basename(resume_path).split('.')[0].split('_')[-1]
        task_name = f"(Resume_from_ep{ep}){task_name}"
    if use_aug:
        task_name += "_Aug"
    if run_type.lower() == "debug":
        task_name = f"(Debug){task_name}"
        epochs = 3 

    # 确定输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir_name = task_name if base_task_name else f'task_{timestamp}'
    task_dir = create_unique_dir(os.path.join(PROJECT_ROOT, 'saved_results', target_dir_name))

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(run_type, use_aug, PROJECT_ROOT)
    
    train_model(
        model=model, criterion=criterion, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader,
        num_epochs=epochs, device=device, task_dir=task_dir,
        resume_path=resume_path, seed=seed, task_name=task_name
    )


# ==========================================
# 4. 动态任务队列管理器 (Task Queue Manager)
# 包含原子写入安全机制。
# ==========================================
def get_next_task(queue_file, history_file):
    if not os.path.exists(queue_file):
        return None
    
    try:
        with open(queue_file, 'r', encoding='utf-8') as f:
            tasks = yaml.safe_load(f)
            
        if not tasks or len(tasks) == 0:
            return None
            
        current_task = tasks.pop(0)
        
        tmp_q_path = f"{queue_file}.tmp"
        with open(tmp_q_path, 'w', encoding='utf-8') as f:
            yaml.dump(tasks, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp_q_path, queue_file)

        archive_task = current_task.copy()
        archive_task["_executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as hf:
                    loaded = yaml.safe_load(hf)
                    if loaded: history = loaded
            except yaml.YAMLError:
                pass
                
        history.append(archive_task)
        
        tmp_h_path = f"{history_file}.tmp"
        with open(tmp_h_path, 'w', encoding='utf-8') as hf:
            yaml.dump(history, hf, allow_unicode=True, sort_keys=False)
        os.replace(tmp_h_path, history_file)
            
        return current_task
        
    except Exception as e:
        print(f"⚠️ 读取任务队列异常: {e}，将暂停 10 秒后重试...")
        time.sleep(10)
        return get_next_task(queue_file, history_file)


if __name__ == '__main__':
    # 路径管理依赖全局的 PROJECT_ROOT
    QUEUE_FILE = os.path.join(PROJECT_ROOT, "logs", "tasks.yml")
    HISTORY_FILE = os.path.join(PROJECT_ROOT, "logs", "tasks_history.yml")
    
    logs_dir = os.path.dirname(QUEUE_FILE)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    if not os.path.exists(QUEUE_FILE):
        tasks_yml_template = """# ==========================================
# 实验任务配置清单 (YAML格式)
# ==========================================
- run_type: debug
  crit: dice
  seed: 48
  use_aug: False
  epochs: 3

- run_type: train_with_pt
  crit: tversky_hd
  seed: 48
  epochs: 50
  loss_params: {'tversky_hd': {'alpha': 0.3, 'beta': 0.7, 'w_tv': 0.8, 'w_hd': 0.2}}
"""
        with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
            f.write(tasks_yml_template)
        print(f"📝 未检测到 {QUEUE_FILE}，已自动生成模板文件！请修改后重新运行代码。")
        exit(0)

    print(f"📂 任务监控已启动。执行历史将自动归档至: {HISTORY_FILE}")
    while True:
        task_kwargs = get_next_task(QUEUE_FILE, HISTORY_FILE)
        
        if task_kwargs is None:
            print("\n✅ 所有任务已执行完毕！任务队列为空。")
            break
            
        print("\n" + "="*50)
        print("▶️ 加载到新任务，参数如下：")
        for k, v in task_kwargs.items():
            print(f"   {k}: {v}")
        print("="*50 + "\n")
        
        main(**task_kwargs)
        time.sleep(1)