"""
MoCaf-Mamba Evaluation Script

Evaluates a model checkpoint on test set with 7 missing modality patterns
and reports Dice, IoU, and class-wise scores.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import warnings

from data_set import MVClaDataset_k
from model.mocaf_mamba import Model


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
    mean_iou = np.mean(ious)
    gdth_sum = np.sum(gts, axis=(2, 3, 4))
    pred_sum = np.sum(preds, axis=(2, 3, 4))
    dice_matrix_sample = 2 * intersection / (gdth_sum + pred_sum + smooth)
    class_wise_dice = np.mean(dice_matrix_sample, axis=0)
    mean_dice = np.mean(dice_matrix_sample)
    return mean_iou, mean_dice, class_wise_dice


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


CASE_NAMES = [
    "Full (111)",
    "Miss Mod3 (110)",
    "Miss Mod2 (101)",
    "Miss Mod1 (011)",
    "Only Mod1 (100)",
    "Only Mod2 (010)",
    "Only Mod3 (001)",
]


def main():
    parser = argparse.ArgumentParser(description="MoCaf-Mamba Model Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Path to dataset directory')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for cross-validation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--num_cls', type=int, default=3,
                        help='Number of segmentation classes')
    parser.add_argument('--num_modals', type=int, default=3,
                        help='Number of input modalities')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    warnings.filterwarnings("ignore")

    device = torch.device('cuda')

    print(f"\n{'='*80}")
    print(f"MoCaf-Mamba Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir:   {args.data_dir}")
    print(f"{'='*80}")

    model = Model(num_cls=args.num_cls, num_modals=args.num_modals)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    test_data = MVClaDataset_k(args.data_dir, 'test', args.fold)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    metrics_recorder = {n: {'dice_all': [], 'iou_all': [], 'dice_cls': []} for n in range(7)}

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
            images, mask_gt, *_ = data
            size = data[-1]
            if isinstance(size, (list, tuple)):
                size = size[0]
            orig_size = size[0].tolist() if torch.is_tensor(size) else size

            images = images.to(device)
            mask_gt = mask_gt.to(device)
            batch_sz = images.size(0)

            for n in range(7):
                mask_input = get_val_mask(n, batch_sz).to(device)
                output = model(images, mask_input)
                seg_pred = F.interpolate(output, size=orig_size, mode='trilinear', align_corners=True)
                seg_pred_idx = torch.argmax(seg_pred, dim=1)
                seg_pred_onehot = F.one_hot(seg_pred_idx, num_classes=args.num_cls).permute(0, 4, 1, 2, 3)
                iou, dice, dice_cls = metric(mask_gt, seg_pred_onehot)
                metrics_recorder[n]['dice_all'].append(dice)
                metrics_recorder[n]['iou_all'].append(iou)
                metrics_recorder[n]['dice_cls'].append(dice_cls)

    # ---- Print Results ----
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Case':<22} | {'Dice':<10} {'IoU':<10} | {'Cls1':<10} {'Cls2':<10}")
    print("-" * 60)

    full_dice = 0.0
    missing_dices = []

    for n in range(7):
        m_dice = np.mean(metrics_recorder[n]['dice_all'])
        m_iou = np.mean(metrics_recorder[n]['iou_all'])
        cls_dices = np.mean(np.array(metrics_recorder[n]['dice_cls']), axis=0)
        print(f"{CASE_NAMES[n]:<22} | {m_dice:.4f}     {m_iou:.4f}     | {cls_dices[0]:.4f}     {cls_dices[1]:.4f}")
        if n == 0:
            full_dice = m_dice
        else:
            missing_dices.append(m_dice)

    print("-" * 60)
    print(f"{'Overall Full Dice':<22} | {full_dice:.4f}")
    print(f"{'Overall Missing Dice':<22} | {np.mean(missing_dices):.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
