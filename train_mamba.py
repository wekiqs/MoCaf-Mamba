"""
MoCaf-Mamba Training Script (DDP)

Distributed training with mixed precision, EMA, warmup + cosine LR scheduling.
"""

import os
import argparse

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from tqdm import tqdm
from data_set import *
import matplotlib.pyplot as plt
import numpy as np
from model.utils import criterions
import warnings
from torch.cuda.amp import GradScaler
from model.mocaf_mamba import Model

torch.manual_seed(3407)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    pred = pred[:, 1:, :, :, :]
    target = target[:, 1:, :, :, :]
    intersection = (pred * target).sum(dim=(2, 3, 4))
    pred_sum = (pred ** 2).sum(dim=(2, 3, 4))
    target_sum = (target ** 2).sum(dim=(2, 3, 4))

    dice = (2. * intersection + eps) / (pred_sum + target_sum + eps)
    loss = 1 - dice
    return loss.mean()


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def metric(gt, pred, smooth=0.001):
    gts = gt.detach().cpu().numpy()
    preds = pred.detach().cpu().numpy()

    gts = gts.astype(int)
    preds = preds.astype(int)

    gts = gts[:, 1:, :, :, :]
    preds = preds[:, 1:, :, :, :]

    intersection = np.sum(np.logical_and(gts, preds), axis=(2, 3, 4))
    union = np.sum(np.logical_or(gts, preds), axis=(2, 3, 4))

    ious = intersection / (union + smooth)
    ious = np.mean(ious)

    gdth_sum = np.sum(gts, axis=(2, 3, 4))
    pred_sum = np.sum(preds, axis=(2, 3, 4))

    dice_matrix = 2 * intersection / (gdth_sum + pred_sum + smooth)
    dices = np.mean(dice_matrix)
    return ious, dices, dice_matrix


def get_training_mask(image, drop_prob=0.2):
    B, C, _, _, _ = image.shape
    mask = torch.rand((B, C), device=image.device) > drop_prob
    keep_counts = mask.sum(dim=1)
    all_dropped_indices = torch.where(keep_counts == 0)[0]
    if len(all_dropped_indices) > 0:
        force_indices = torch.randint(0, C, (len(all_dropped_indices),), device=image.device)
        mask[all_dropped_indices, force_indices] = True
    return mask.bool()


def get_val_mask(n, B):
    all_mask = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    mask = all_mask[n]
    mask = torch.tensor(mask, dtype=torch.bool)
    mask = mask.repeat(B, 1)
    return mask.bool()


def main():
    parser = argparse.ArgumentParser(description="MoCaf-Mamba Training (DDP)")
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--task', type=str, default='mocaf_mamba',
                        help='Task name for checkpoint subdirectory')
    parser.add_argument('--fold', type=int, default=0,
                        help='Cross-validation fold index')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_cls', type=int, default=3,
                        help='Number of segmentation classes')
    parser.add_argument('--num_modals', type=int, default=3,
                        help='Number of input modalities')
    args = parser.parse_args()

    # ---------- DDP init ----------
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    warnings.filterwarnings("ignore")

    task_dir = os.path.join(args.output_dir, args.task)
    fold_dir = os.path.join(task_dir, str(args.fold))
    os.makedirs(fold_dir, exist_ok=True)

    if rank == 0:
        print(f'Task: {args.task}')
        print(f'Data dir: {args.data_dir}')
        print(f'Output dir: {args.output_dir}')
        print(f'Fold: {args.fold}')
        print(f'Epochs: {args.epochs}, LR: {args.lr}')
        print(f'World size: {world_size}')

    model = Model(num_cls=args.num_cls, num_modals=args.num_modals)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=3e-5, betas=(0.9, 0.999))

    ema = EMAModel(model.module, decay=0.999)
    ema.register()

    train_data = MVClaDataset_k(args.data_dir, 'train', args.fold)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
        sampler=train_sampler, multiprocessing_context='spawn', prefetch_factor=4, drop_last=True,
    )

    valid_data = MVClaDataset_k(args.data_dir, 'test', args.fold)
    valid_sampler = DistributedSampler(valid_data, num_replicas=world_size, rank=rank, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        sampler=valid_sampler, drop_last=False,
    )

    warmup_epochs = 5
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=2e-6),
        ],
        milestones=[warmup_epochs],
    )

    scaler = GradScaler()

    if rank == 0:
        pbar = tqdm(total=args.epochs)
        pbar.set_description('Training')
        pbar.set_postfix(loss='0.000')
    else:
        pbar = None

    best_dice_missing = 0.5
    best_dice_full = 0.5
    e_loss = {
        'fuse_cross_loss': [], 'fuse_dice_loss': [], 'sep_cross_loss': [], 'sep_dice_loss': [],
        'prm_cross_loss': [], 'prm_dice_loss': [], 'sim_loss': [], 'flow_loss': [],
    }
    e_val = {'dice_full': [], 'dice_missing': [], 'loss': []}

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        model.module.is_training = True
        b_loss = {
            'fuse_cross_loss': [], 'fuse_dice_loss': [], 'sep_cross_loss': [], 'sep_dice_loss': [],
            'prm_cross_loss': [], 'prm_dice_loss': [], 'sim_loss': [], 'flow_loss': [],
        }
        num_cls = args.num_cls
        torch.cuda.empty_cache()

        for i, data in enumerate(train_loader):
            images, mask, label_d = data
            torch.set_grad_enabled(True)
            images, mask, label_d = images.to(device), mask.to(device), label_d.to(device)
            target = mask
            mask_ = get_training_mask(images, drop_prob=0.5).to(device)
            fuse_pred, sep_preds, prm_preds, sim_loss = model(images, mask_)
            sim_loss = sim_loss.mean()

            fuse_cross_loss = criterions.softmax_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = dice_loss(fuse_pred, target)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).to(device).float()
            sep_dice_loss = torch.zeros(1).to(device).float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += dice_loss(sep_pred, target)
            sep_loss = sep_cross_loss + sep_dice_loss
            sep_loss /= len(sep_preds)

            prm_cross_loss = torch.zeros(1).to(device).float()
            prm_dice_loss = torch.zeros(1).to(device).float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += dice_loss(prm_pred, target)
            prm_loss = prm_cross_loss + prm_dice_loss
            prm_loss /= len(prm_preds)

            loss = fuse_loss + sep_loss + prm_loss + sim_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            ema.update()

            b_loss['fuse_cross_loss'].append(fuse_cross_loss.item())
            b_loss['fuse_dice_loss'].append(fuse_dice_loss.item())
            b_loss['sep_cross_loss'].append(sep_cross_loss.item())
            b_loss['sep_dice_loss'].append(sep_dice_loss.item())
            b_loss['prm_cross_loss'].append(prm_cross_loss.item())
            b_loss['prm_dice_loss'].append(prm_dice_loss.item())
            b_loss['sim_loss'].append(sim_loss.item())
            b_loss['flow_loss'].append(0.0)

        scheduler.step()

        if rank == 0:
            pbar.update(1)
            pbar.set_postfix(loss='{:.6f}'.format(np.mean(b_loss['fuse_dice_loss'])))
            print(
                f"fuse_dice_loss{np.mean(b_loss['fuse_dice_loss'])},"
                f"sep_dice_loss{np.mean(b_loss['sep_dice_loss'])},"
                f"prm_dice_loss{np.mean(b_loss['prm_dice_loss'])}"
            )
            print(
                f"fuse_cross_loss{np.mean(b_loss['fuse_cross_loss'])},"
                f"sep_cross_loss{np.mean(b_loss['sep_cross_loss'])},"
                f"prm_cross_loss{np.mean(b_loss['prm_cross_loss'])}"
            )
            print(f"sim_loss{np.mean(b_loss['sim_loss'])},flow_loss{np.mean(b_loss['flow_loss'])}")

            for k in b_loss.keys():
                e_loss[k].append(np.mean(b_loss[k]))
            for k in e_loss.keys():
                plt.plot(e_loss[k], label=k)
            plt.legend()
            plt.ylim(0, 1)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(fold_dir, 'loss.png'))
            plt.close()

        # Validate: epoch >= 200, every 5 epochs
        should_validate = False
        if epoch >= 200 and epoch % 5 == 0:
            local_fuse_dice = np.mean(b_loss['fuse_dice_loss'])
            local_tensor = torch.tensor(local_fuse_dice, device=device)
            dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
            global_fuse_dice = local_tensor.item() / world_size
            if rank == 0:
                should_validate = global_fuse_dice <= 0.4
            should_validate_tensor = torch.tensor(1 if should_validate else 0, device=device)
            dist.broadcast(should_validate_tensor, src=0)
            should_validate = should_validate_tensor.item() == 1

        if should_validate:
            with torch.no_grad():
                ema.apply_shadow()
                model.eval()

                b_loss_test = torch.tensor(0.0, device=device)
                b_dice_full = torch.tensor(0.0, device=device)
                b_dice_missing = torch.tensor(0.0, device=device)
                b_count = torch.tensor(0.0, device=device)

                for i, data in enumerate(valid_loader):
                    images_t, mask_t, *_ = data
                    images_t, mask_t = images_t.to(device, non_blocking=True), mask_t.to(device, non_blocking=True)
                    size_t = data[-1]
                    if isinstance(size_t, (list, tuple)):
                        size_t = size_t[0]
                    orig_size = size_t[0].tolist() if torch.is_tensor(size_t) else size_t

                    batch_sz = images_t.size(0)

                    for n in range(7):
                        mask_ = get_val_mask(n, batch_sz).to(images_t.device)
                        fuse_pred = model(images_t, mask_)
                        fuse_pred_resized = F.interpolate(fuse_pred, size=orig_size, mode='trilinear', align_corners=True)

                        loss = dice_loss(fuse_pred_resized, mask_t)
                        pre_s = F.one_hot(fuse_pred_resized.argmax(dim=1), num_classes=num_cls).permute(0, 4, 1, 2, 3)
                        iou, dice, dice_matrix = metric(mask_t, pre_s)

                        b_loss_test += loss.item() * batch_sz

                        if n == 0:
                            b_dice_full += dice * batch_sz
                        else:
                            b_dice_missing += dice * batch_sz

                    b_count += batch_sz

                dist.all_reduce(b_loss_test, op=dist.ReduceOp.SUM)
                dist.all_reduce(b_dice_full, op=dist.ReduceOp.SUM)
                dist.all_reduce(b_dice_missing, op=dist.ReduceOp.SUM)
                dist.all_reduce(b_count, op=dist.ReduceOp.SUM)

                if rank == 0:
                    global_loss = b_loss_test.item() / (b_count.item() * 7)
                    global_dice_full = b_dice_full.item() / b_count.item()
                    global_dice_missing = b_dice_missing.item() / (b_count.item() * 6)
                    e_val['dice_full'].append(global_dice_full)
                    e_val['dice_missing'].append(global_dice_missing)
                    e_val['loss'].append(global_loss)

                    print(f'Validation - Loss: {global_loss:.4f}')
                    print(f'Validation - Dice (Full Modality): {global_dice_full:.4f}')
                    print(f'Validation - Dice (Missing Avg): {global_dice_missing:.4f}')

                    plt.clf()
                    plt.ylim(0, 1)
                    plt.plot(e_val['dice_full'], label='Dice Full')
                    plt.plot(e_val['dice_missing'], label='Dice Missing Avg')
                    plt.plot(e_val['loss'], label='Loss')
                    plt.legend()
                    plt.xlabel('Epoch')
                    plt.ylabel('Score')
                    plt.savefig(os.path.join(fold_dir, 'val_metrics.png'))
                    plt.close()

                    if global_dice_full >= best_dice_full:
                        best_dice_full = global_dice_full
                        torch.save(
                            model.module.state_dict(),
                            os.path.join(fold_dir, f'{epoch:03d}full_dice_{global_dice_full:.4f}_missing_dice_{global_dice_missing:.4f}.pth'),
                        )
                        print(f'>>> Saved BEST FULL Model with dice: {global_dice_full:.4f}')

                    if global_dice_missing >= best_dice_missing:
                        best_dice_missing = global_dice_missing
                        torch.save(
                            model.module.state_dict(),
                            os.path.join(fold_dir, f'{epoch:03d}full_dice_{global_dice_full:.4f}_missing_dice_{global_dice_missing:.4f}.pth'),
                        )
                        print(f'>>> Saved BEST MISSING Model with dice: {global_dice_missing:.4f}')

                    torch.save(model.module.state_dict(), os.path.join(fold_dir, 'last.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(fold_dir, 'optimizer.pth'))

                ema.restore()

    if rank == 0:
        pbar.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
