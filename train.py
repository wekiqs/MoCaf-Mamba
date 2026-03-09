import argparse
import os
import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torchvision

from data_set_nz import MVClaDataset_k
from model_seg.moca_net import Moca_net
from model_seg.utils import criterions

warnings.filterwarnings("ignore")
torchvision.disable_beta_transforms_warning()


# ---------------------- 辅助函数 ----------------------

def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=datetime.timedelta(seconds=7200)
        )
    else:
        # 非分布式环境的回退（单卡调试用）
        rank = 0
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(0)
        print("Warning: Distributed environment not detected. Running in single process mode.")

    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def dist_print(msg, rank):
    """仅在主进程打印"""
    if rank == 0:
        print(msg)


def reduce_tensor(tensor):
    """全归约：求平均"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def set_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------- Loss & Metrics ----------------------

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    # 假设 pred 是 softmax 后的概率 [B, C, D, H, W]
    # 跳过背景类 (index 0)
    pred = pred[:, 1:, ...]
    target = target[:, 1:, ...]

    intersection = (pred * target).sum(dim=(2, 3, 4))
    pred_sum = (pred ** 2).sum(dim=(2, 3, 4))
    target_sum = (target ** 2).sum(dim=(2, 3, 4))

    dice = (2. * intersection + eps) / (pred_sum + target_sum + eps)
    return (1 - dice).mean()


def get_training_mask(image, drop_prob=0.5):
    B, C, _, _, _ = image.shape
    mask = torch.rand((B, C), device=image.device) > drop_prob
    keep_counts = mask.sum(dim=1)
    all_dropped_indices = torch.where(keep_counts == 0)[0]
    if len(all_dropped_indices) > 0:
        force_indices = torch.randint(0, C, (len(all_dropped_indices),), device=image.device)
        mask[all_dropped_indices, force_indices] = True
    return mask.bool()


def get_val_mask(n, B, device):
    all_mask = [
        [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0],
        [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0],
        [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0],
        [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]
    ]
    mask = torch.tensor(all_mask[n], dtype=torch.bool, device=device)
    mask = mask.repeat(B, 1)
    return mask


def metric_calc(gt, pred, smooth=0.001):
    """计算单个batch的指标"""
    # 移至 CPU 计算以兼容 numpy 操作，或者重写为纯 torch 操作以加速
    gts = gt.detach().cpu().numpy().astype(int)
    preds = pred.detach().cpu().numpy().astype(int)

    # 逻辑调整：根据原始代码 [1, 3, 2] 通道置换
    gts = gts[:, [1, 3, 2], :, :, :]
    preds = preds[:, [1, 3, 2], :, :, :]

    # 包含关系处理
    gts[:, 1][gts[:, 0] >= 0.5] = 1
    gts[:, 2][gts[:, 1] >= 0.5] = 1
    preds[:, 1][preds[:, 0] >= 0.5] = 1
    preds[:, 2][preds[:, 1] >= 0.5] = 1

    intersection = np.sum(np.logical_and(gts, preds), axis=(2, 3, 4))
    union = np.sum(np.logical_or(gts, preds), axis=(2, 3, 4))

    # Dice
    gdth_sum = np.sum(gts, axis=(2, 3, 4))
    pred_sum = np.sum(preds, axis=(2, 3, 4))

    # 计算 Dice
    dice_val = (2 * intersection) / (gdth_sum + pred_sum + smooth)
    return np.mean(dice_val)  # 返回该 Batch 所有样本和类别的平均 Dice


# ---------------------- 核心流程 ----------------------

def train_one_epoch(model, loader, optimizer, epoch, scaler, rank, args):
    model.train()
    # 如果模型里有自定义的 is_training 标志
    if hasattr(model.module, 'is_training'):
        model.module.is_training = True

    num_cls = 4
    loss_meter = {
        'fuse_dice': [], 'sep_dice': [], 'prm_dice': [],
        'fuse_cross': [], 'sep_cross': [], 'prm_cross': [],
        'sim': []
    }

    pbar = None
    if rank == 0:
        pbar = tqdm(total=len(loader), desc=f'Train Ep {epoch}', leave=False)

    for data in loader:
        images, mask, label_d = data
        images = images.cuda(args.local_rank, non_blocking=True)
        mask = mask.cuda(args.local_rank, non_blocking=True)
        # label_d = label_d.cuda(args.local_rank, non_blocking=True) # 如未使用可注释

        target = mask
        mask_drop = get_training_mask(images, drop_prob=0.5)

        optimizer.zero_grad()

        # 混合精度上下文
        with autocast(enabled=True):  # 建议启用 AMP
            fuse_pred, sep_preds, prm_preds, sim_loss = model(images, mask_drop)

            # 1. Fuse Loss
            fuse_cross = criterions.softmax_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice = dice_loss(fuse_pred, target)
            fuse_loss = fuse_dice + fuse_cross

            # 2. Sep Loss
            sep_loss = 0
            sep_cross_accum, sep_dice_accum = 0, 0
            for p in sep_preds:
                c_loss = criterions.softmax_loss(p, target, num_cls=num_cls)
                d_loss = dice_loss(p, target)
                sep_loss += (c_loss + d_loss)
                sep_cross_accum += c_loss
                sep_dice_accum += d_loss
            if len(sep_preds) > 0:
                sep_loss /= len(sep_preds)
                sep_cross_accum /= len(sep_preds)
                sep_dice_accum /= len(sep_preds)

            # 3. Prm Loss
            prm_loss = 0
            prm_cross_accum, prm_dice_accum = 0, 0
            for p in prm_preds:
                c_loss = criterions.softmax_loss(p, target, num_cls=num_cls)
                d_loss = dice_loss(p, target)
                prm_loss += (c_loss + d_loss)
                prm_cross_accum += c_loss
                prm_dice_accum += d_loss
            if len(prm_preds) > 0:
                prm_loss /= len(prm_preds)
                prm_cross_accum /= len(prm_preds)
                prm_dice_accum /= len(prm_preds)

            total_loss = fuse_loss + sep_loss + prm_loss + sim_loss

        # 检查 NaN
        if torch.isnan(total_loss):
            dist_print(f"NAN Loss detected at epoch {epoch}", rank)
            continue

        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()

        # 记录日志 (Reduce 只用于显示，不影响梯度)
        loss_dict = {
            'fuse_dice': fuse_dice, 'sep_dice': sep_dice_accum, 'prm_dice': prm_dice_accum,
            'fuse_cross': fuse_cross, 'sep_cross': sep_cross_accum, 'prm_cross': prm_cross_accum,
            'sim': sim_loss
        }

        # 仅在 Rank 0 收集用于打印
        if rank == 0:
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    loss_meter[k].append(v.item())
                else:
                    loss_meter[k].append(v)
            pbar.update(1)
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    if rank == 0:
        pbar.close()
        # 计算 Epoch 平均 Loss
        avg_losses = {k: np.mean(v) for k, v in loss_meter.items()}
        print(f"\nEpoch {epoch} Summary:")
        print(f"Dice Loss - Fuse: {avg_losses['fuse_dice']:.4f}, Sep: {avg_losses['sep_dice']:.4f}, Prm: {avg_losses['prm_dice']:.4f}")
        print(f"Sim Loss: {avg_losses['sim']:.4f}")
        return avg_losses
    return None


def validate(model, loader, epoch, rank, args):
    model.eval()
    if hasattr(model.module, 'is_training'):
        model.module.is_training = False

    metrics = {
        'dice_full': torch.tensor(0.0).cuda(args.local_rank),
        'dice_missing': torch.tensor(0.0).cuda(args.local_rank),
        'count': torch.tensor(0.0).cuda(args.local_rank)
    }

    with torch.no_grad():
        for data in loader:
            images, mask, _ = data
            images = images.cuda(args.local_rank, non_blocking=True)
            mask_target = mask.cuda(args.local_rank, non_blocking=True)  # [B, C, D, H, W]

            batch_sz = images.size(0)

            # 遍历15种模态组合
            # n=0: 全模态, n=1~14: 缺失模态
            for n in range(15):
                mask_in = get_val_mask(n, batch_sz, images.device)
                pred_logit = model(images, mask_in)

                # 计算 Dice
                # 需要做 Argmax 和 One-hot 转换以匹配 metric_calc 的输入
                pred_idx = pred_logit.argmax(dim=1)
                pred_onehot = F.one_hot(pred_idx, num_classes=4).permute(0, 4, 1, 2, 3)

                # metric_calc 返回的是 numpy float，需要转回 tensor 累加
                dice_val = metric_calc(mask_target, pred_onehot)
                dice_tensor = torch.tensor(dice_val).cuda(args.local_rank)

                if n == 0:
                    metrics['dice_full'] += dice_tensor * batch_sz
                else:
                    metrics['dice_missing'] += dice_tensor * batch_sz

            metrics['count'] += batch_sz

    # 分布式汇总
    for k in metrics:
        metrics[k] = reduce_tensor(metrics[k])

    global_count = metrics['count'].item()
    if global_count == 0: return 0, 0

    # missing 计算了 14 种情况，full 计算了 1 种
    dice_full = metrics['dice_full'].item() / global_count
    dice_missing = metrics['dice_missing'].item() / (global_count * 14)

    dist_print(f">>> Validation Ep {epoch}: Full Dice: {dice_full:.4f} | Missing Avg Dice: {dice_missing:.4f}", rank)

    return dice_full, dice_missing


# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument('--gpu_ids', type=str, default=None, help='Deprecated. Use CUDA_VISIBLE_DEVICES via shell.')
    parser.add_argument('--data_root', type=str, default='../data/classification_nii_pz', help='Path to dataset')
    parser.add_argument('--fold', type=int, default=0, help='Cross validation fold')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)  # per gpu
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--task_name', type=str, default='t_4_')

    # 恢复训练
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--load_optim', action='store_true', help='Load optimizer state if resuming')

    # 系统参数
    # parser.add_argument('--local_rank', type=int, default=0) # torchrun 会自动处理

    args = parser.parse_args()

    # 1. 设置 DDP
    rank, local_rank, world_size = setup_distributed()
    args.local_rank = local_rank  # 存入 args 方便传递
    set_seed(3407 + rank)

    save_path = os.path.join(args.save_dir, args.task_name, str(args.fold))
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        print(f"Training Config: {args}")

    # 2. 数据集
    train_data = MVClaDataset_k(args.data_root, 'train', args.fold)
    valid_data = MVClaDataset_k(args.data_root, 'test', args.fold)

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_data, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=8, pin_memory=True, prefetch_factor=4, drop_last=True
    )
    # 验证集 Batch Size 可以稍微大一点
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=4, sampler=valid_sampler,
        num_workers=4, pin_memory=True
    )

    # 3. 模型
    model = Moca_net(num_cls=4)
    model.cuda(local_rank)

    # 4. 优化器 & 调度器
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=3e-5, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-6)
    scaler = GradScaler()

    # 5. 加载权重 (Resume)
    start_epoch = 0
    best_dice_full = 0.0

    if args.resume and os.path.exists(args.resume):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.resume, map_location=map_location)

        # 兼容只保存了 state_dict 或保存了完整 checkpoint 字典的情况
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.load_optim and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_dice_full = checkpoint.get('best_dice', 0.0)
                if 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
            dist_print(f"Resumed from {args.resume}, start epoch {start_epoch}", rank)
        else:
            # 假设直接保存的是 state_dict
            model.load_state_dict(checkpoint, strict=False)
            dist_print(f"Loaded weights from {args.resume} (weights only)", rank)

    # 包装 DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 绘图数据
    log_history = {'loss': [], 'dice_full': [], 'dice_missing': []}

    # 6. 循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # 训练
        train_metrics = train_one_epoch(model, train_loader, optimizer, epoch, scaler, rank, args)
        scheduler.step()

        # 记录训练 Loss
        if rank == 0 and train_metrics:
            log_history['loss'].append(train_metrics.get('fuse_dice', 0))

        # 验证 (每2轮且 epoch>=4)
        if (epoch % 2 == 0 and epoch >= 4) or epoch == args.epochs - 1:
            d_full, d_miss = validate(model, valid_loader, epoch, rank, args)

            if rank == 0:
                log_history['dice_full'].append(d_full)
                log_history['dice_missing'].append(d_miss)

                # 绘图
                plt.figure()
                plt.plot(log_history['dice_full'], label='Dice Full')
                plt.plot(log_history['dice_missing'], label='Dice Missing')
                plt.legend()
                plt.title('Validation Dice')
                plt.savefig(os.path.join(save_path, 'val_metrics.png'))
                plt.close()

                # 保存 Checkpoint
                state_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_dice': best_dice_full
                }

                # 保存最新的
                torch.save(state_dict, os.path.join(save_path, 'last.pth'))

                # 保存最佳的 Full Dice
                if d_full > best_dice_full:
                    best_dice_full = d_full
                    torch.save(state_dict, os.path.join(save_path, f'best_full_{d_full:.4f}.pth'))
                    print(f"Saved Best Full Model: {d_full:.4f}")

    cleanup_distributed()


if __name__ == '__main__':
    main()
