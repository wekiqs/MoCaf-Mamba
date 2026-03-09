import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import warnings

# 引入项目依赖 (请确保路径正确)
from data_set_nz import MVClaDataset_k
from model_seg.moca_net import Moca_net as Model

# ---------------- 配置区域 ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
BATCH_SIZE = 4  # 必须为 1 才能逐个样本统计名称
NUM_WORKERS = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def metric(gt, pred, smooth=0.001, n=None):
    """
    完全在 GPU 上计算指标，避免 IO 瓶颈。
    gt, pred: Tensor [B, 4, D, H, W] (One-hot 格式)
    """
    with torch.no_grad():
        # 提取通道 [B, 3, D, H, W] (对应 WT, TC, ET 或 Class 1, 2, 3)
        gt_sub = gt[:, [1, 3, 2], :, :, :].float()
        pred_sub = pred[:, [1, 3, 2], :, :, :].float()

        # 区域包含逻辑 (BraTS 常用逻辑: ET包含于TC, TC包含于WT)
        gt_sub[:, 1] = torch.max(gt_sub[:, 1], gt_sub[:, 0])
        gt_sub[:, 2] = torch.max(gt_sub[:, 2], gt_sub[:, 1])

        pred_sub[:, 1] = torch.max(pred_sub[:, 1], pred_sub[:, 0])
        pred_sub[:, 2] = torch.max(pred_sub[:, 2], pred_sub[:, 1])

        # Intersection
        intersection = torch.sum(gt_sub * pred_sub, dim=(2, 3, 4))
        gt_sum = torch.sum(gt_sub, dim=(2, 3, 4))
        pred_sum = torch.sum(pred_sub, dim=(2, 3, 4))

        ious_matrix = (intersection + smooth) / (gt_sum + pred_sum - intersection + smooth)

        # 1. 正常计算 Dice (包含 smooth 防止除零)
        dice_matrix = (2 * intersection + smooth) / (gt_sum + pred_sum + smooth)

        # --- 修正逻辑开始 ---

        # 判断 GT 和 Pred 是否为空
        gt_empty = gt_sum <= 64
        pred_empty = pred_sum <= 64

        # 情况 A: GT 为空 且 Pred 为空 -> 预测完美，Dice 置为 1
        # (注意：上面的公式在 smooth 很小时，0/smooth 接近 0，所以需要手动置 1)
        true_negative = gt_empty & pred_empty
        dice_matrix[true_negative] = 1.0

        # 情况 B: GT 为空 但 Pred 不为空 -> 假阳性，Dice 置为 0
        # (上面的公式 2*0 / (0 + pred + sm) 已经是 0 了，通常不需要额外操作，
        #  但如果 smooth 很大可能会有偏差，不过通常保持公式计算结果即可，或强制置 0)
        false_positive = gt_empty & (~pred_empty)
        dice_matrix[false_positive] = 0.0

        # 情况 C: GT 不为空 -> 正常计算 (上面公式已涵盖)

        # --- 修正逻辑结束 ---

        # 聚合统计
        mean_dice = torch.mean(dice_matrix).item()
        class_dice = torch.mean(dice_matrix, dim=0).cpu().numpy()
        mean_iou = torch.mean(ious_matrix).item()

        # if class_dice[0] < 0.2 and n == 0:
        #     print("Low Dice:", class_dice)

        return mean_iou, mean_dice, class_dice


def get_val_mask(n, B):
    all_mask = [
        [1, 1, 1, 1],  # 0: Full Modality
        [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
        [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ]
    mask = all_mask[n]
    mask = torch.tensor(mask, dtype=torch.bool)
    mask = mask.unsqueeze(0).repeat(B, 1)
    return mask


def test(fold, ckpt_path):
    print(f"Testing Fold: {fold}")
    print(f"Loading weights from: {ckpt_path}")

    # 1. 初始化模型
    model = Model(num_cls=4, num_modals=4)

    # 2. 加载权重
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()

    # 3. 数据加载器
    test_data = MVClaDataset_k('../data/classification_nii_pz', 'test', fold)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4)

    class_names = ["Class_1", "Class_2", "Class_3"]

    # 全局指标记录器 (按模态组合)
    metrics_recorder = {
        case_id: {'dice_all': [], 'iou_all': [], 'dice_cls': []} for case_id in range(15)
    }

    # 【新增】样本级指标记录器 (按样本名)
    sample_wise_stats = {}

    case_names = [
        "Full (1111)", "Miss Mod4 (1110)", "Miss Mod3 (1101)", "Miss Mod2 (1011)", "Miss Mod1 (0111)",
        "Only Mod1,2 (1100)", "Only Mod1,3 (1010)", "Only Mod1,4 (1001)", "Only Mod2,3 (0110)",
        "Only Mod2,4 (0101)", "Only Mod3,4 (0011)", "Only Mod1 (1000)", "Only Mod2 (0100)",
        "Only Mod3 (0010)", "Only Mod4 (0001)",
    ]

    # 4. 推理循环
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference"):
            images, mask_gt, name, size = data

            sample_id = name[0]

            images = images.to(device)
            mask_gt = mask_gt.to(device)
            current_batch_size = images.size(0)

            # 临时列表：存储当前样本在 15 种模态下的 class dice
            # 维度将是 [15, 3]
            current_sample_cls_dices = []

            # 遍历所有 15 种模态组合
            for n in range(15):
                mask_input = get_val_mask(n, current_batch_size).to(device)  # 注意：这里不需要*2，除非你有flip augmentation

                # 模型推理
                output = model(images, mask_input)
                seg_pred = output

                # Argmax & One-hot
                seg_pred_idx = torch.argmax(seg_pred, dim=1)
                seg_pred_onehot = F.one_hot(seg_pred_idx, num_classes=4).permute(0, 4, 1, 2, 3)

                # 计算指标
                iou, dice, dice_cls = metric(mask_gt, seg_pred_onehot, n = n)

                # 记录到全局统计
                metrics_recorder[n]['dice_all'].append(dice)
                metrics_recorder[n]['iou_all'].append(iou)
                metrics_recorder[n]['dice_cls'].append(dice_cls)

                # 【新增】记录当前样本的 class dice
                current_sample_cls_dices.append(dice_cls)

            # 【新增】计算当前样本在所有模态下的平均 Class Dice
            # stack后形状为 (15, 3), 对 axis=0 求平均 -> (3,)
            avg_cls_dice_per_sample = np.mean(np.stack(current_sample_cls_dices), axis=0)
            sample_wise_stats[sample_id] = avg_cls_dice_per_sample

    # -------------------------------------------------------------------------
    # 5. 输出：按样本统计 (Sample-wise Statistics)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Sample ID':<30} | {class_names[0]:<12} | {class_names[1]:<12} | {class_names[2]:<12}")
    print(f"{'(Avg over 15 modalities)':<30} | {'Dice':<12} | {'Dice':<12} | {'Dice':<12}")
    print("-" * 80)

    # for s_id, s_dices in sample_wise_stats.items():
    #     print(f"{s_id:<30} | {s_dices[0]:.4f}       | {s_dices[1]:.4f}       | {s_dices[2]:.4f}")

    print("=" * 80 + "\n")

    # 6. 汇总输出 (原有的 Overall Statistics)
    print("=" * 90)
    print(f"{'Case Name':<25} | {'Mean Dice':<10} | {'Mean IoU':<10} | "
          f"{class_names[0]:<8} | {class_names[1]:<8} | {class_names[2]:<8}")
    print("-" * 90)

    full_dice = 0.0
    miss_one_dices = []
    miss_two_dices = []
    miss_three_dices = []
    class_stats = {i: [] for i in range(3)}

    for n in range(15):
        m_dice = np.mean(metrics_recorder[n]['dice_all'])
        m_iou = np.mean(metrics_recorder[n]['iou_all'])
        cls_dices = np.mean(np.stack(metrics_recorder[n]['dice_cls']), axis=0)

        if n == 0:
            full_dice = m_dice
        elif 1 <= n <= 4:
            miss_one_dices.append(m_dice)
        elif 5 <= n <= 10:
            miss_two_dices.append(m_dice)
        else:
            miss_three_dices.append(m_dice)

        for c in range(3):
            class_stats[c].append(cls_dices[c])

        print(f"{case_names[n]:<25} | {m_dice:.4f}     | {m_iou:.4f}     | "
              f"{cls_dices[0]:.4f}   | {cls_dices[1]:.4f}   | {cls_dices[2]:.4f}")

    print("=" * 90)

    print(f"\n{'=' * 50} Overall Statistics {'=' * 50}")
    print(f"Full Modality Dice:                 {full_dice:.4f}")
    print(f"Missing One Modality Avg Dice:      {np.mean(miss_one_dices):.4f}")
    print(f"Missing Two Modalities Avg Dice:    {np.mean(miss_two_dices):.4f}")
    print(f"Missing Three Modalities Avg Dice:  {np.mean(miss_three_dices):.4f}")
    print(f"Overall Missing Modality Dice:      {np.mean(miss_one_dices + miss_two_dices + miss_three_dices):.4f}")
    print("=" * 90)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    fold = 0
    ckpt_path = '/raid/zhouyongsong/Eye_neoplasms/fuse/pt_mmformer/0/155full_dice_0.8494_missing_dice_0.7809.pth'
    test(fold, ckpt_path)
