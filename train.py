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
                new_dir = base_dir + suffix + str(counter)
            
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                return new_dir
            counter += 1

def validate_model(model, val_loader, criterion, device):
    """单轮验证函数"""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)

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
    if resume_path and os.path.exists(resume_path):
        if resume_path is not None:
            log(f"🔄 Loading checkpoint from: {os.path.abspath(resume_path)}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss) 
        log(f"▶ Resuming training from epoch {start_epoch}")
    else:
        log("🆕 Starting training from scratch")
    # ===================================
    
    model = model.to(device)
    train_losses = []
    start_time = time.time()
    best_epoch = -1
    
    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            log(f'Epoch {epoch+1}/{num_epochs} started')
            
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            val_loss = validate_model(model, val_loader, criterion, device)
            epoch_duration = time.time() - epoch_start
            train_losses.append(epoch_loss)
            
            log(f" → Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Epoch time: {epoch_duration:.2f}s")
            
            # 每10轮或最后一轮保存模型
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                model_path = os.path.join(models_dir, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }, model_path)
                log(f"💾 Saved model: model_epoch_{epoch+1}.pth")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_path = os.path.join(models_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }, best_model_path)
                log(f"🏆 New best model saved (val_loss={best_val_loss:.6f}) → best_model.pth")
    
    except KeyboardInterrupt:
        log("⚠️ Training interrupted by user (Ctrl+C)")
        raise
    except Exception as e:
        log(f"❌ Training failed with error: {str(e)}")
        raise
    
    finally:
                # 保存 loss 曲线
        if train_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
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
        loss_params = {'focal':{'alpha':0.25, 'gamma':2} ,'tversky':{'alpha':0.3, 'beta':0.7}},
        resume_path=None, seed=48, epoch=50, use_aug=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    BASE_DIR = r"d:\Work\Python\_MSDT"
    TASK_NAME = None
    EPOCH = epoch
    set_seed(seed)
    
    #   Adam Vs AdamW
    if opt.lower() in ("adamw", "default"):
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=set_decay)
    elif opt.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=set_decay)
    else:
        optimizer = optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=set_decay)
        print(f"⚠️ Unknown type of optimizer: {opt}, using AdamW as default.")
    if opt.lower() != "default":
        if set_decay == 0:
            TASK_NAME = f"{str(optimizer).split(' (')[0]}_wd_{set_decay}[Seed_{seed}]"
        else:
            TASK_NAME = f"{str(optimizer).split(' (')[0]}_wd_{set_decay:.0e}[Seed_{seed}]"

    #   Loss Functions
    if crit.lower() == "bce" or crit.lower() == "default":
        criterion = nn.BCEWithLogitsLoss()
        TASK_NAME = f"BCEDiceLoss[Seed_{seed}]"
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
        TASK_NAME += "(Debug)"
        EPOCH = 3 
    elif run_type.lower() == "train_with_pt":
        if use_aug:
            PT_TRAIN = r"d:\Work\Python\_MSDT\dataset\train_aug.pt"
            print("Using augmentation, loading train_aug.pt")
        else:
            PT_TRAIN = r"d:\Work\Python\_MSDT\dataset\train.pt"
            print("Using original, loading train.pt")
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
        num_epochs=EPOCH,
        device=device,
        base_dir=BASE_DIR,
        resume_path=resume_path,
        seed=seed,
        task_name=TASK_NAME
    )


if __name__ == '__main__':
    my_loss_param = {
        'focal':{'alpha':0.25, 'gamma':2} 
        ,"tversky":{'alpha':0.3, 'beta':0.7}
        ,"tversky_hd":{'alpha':0.3, 'beta':0.7, 'w_tv':0.8, 'w_hd':0.2}
    }
    # main(run_type="debug", crit="tversky_hd", seed=48, loss_params=my_loss_param)
    # main(opt="adamw", crit="tversky_hd", seed=48, loss_params=my_loss_param)
    # main(opt="adam", crit="bce", seed=48, loss_params=my_loss_param, use_aug=True)
    main(opt="adamw", crit="tversky_hd", seed=48, loss_params={"tversky_hd":{'alpha':0.3, 'beta':0.7, 'w_tv':0.6, 'w_hd':0.4}})
    main(opt="adamw", crit="tversky_hd", seed=48, loss_params={"tversky_hd":{'alpha':0.3, 'beta':0.7, 'w_tv':0.7, 'w_hd':0.3}})
    pass