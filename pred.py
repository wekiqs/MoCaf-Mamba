import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import warnings
import nibabel as nib  # 必须引入 nibabel

# 引入项目依赖
from data_set_nz import MVClaDataset_k
from model_seg.moca_net import Moca_net as Model

# ---------------- 配置区域 ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
# 保存结果必须为 1，以确保文件名和图像一一对应
BATCH_SIZE = 1
NUM_WORKERS = 4
SAVE_DIR = './inference_results'  # 结果保存的根目录

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_val_mask(n, B):
    all_mask = [
        [1, 1, 1, 1],  # 0: Full Modality
        [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],  # Missing 1
        [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],  # Missing 2
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],  # Missing 3
    ]
    mask = all_mask[n]
    mask = torch.tensor(mask, dtype=torch.bool)
    mask = mask.unsqueeze(0).repeat(B, 1)
    return mask


def get_case_name(n):
    """为15种模态组合定义文件夹名称"""
    case_names = [
        "00_Full_1111",
        "01_Miss_Mod4_1110", "02_Miss_Mod3_1101", "03_Miss_Mod2_1011", "04_Miss_Mod1_0111",
        "05_Only_Mod12_1100", "06_Only_Mod13_1010", "07_Only_Mod14_1001",
        "08_Only_Mod23_0110", "09_Only_Mod24_0101", "10_Only_Mod34_0011",
        "11_Only_Mod1_1000", "12_Only_Mod2_0100",
        "13_Only_Mod3_0010", "14_Only_Mod4_0001",
    ]
    return case_names[n]


def save_nifti(pred_numpy, save_path, affine=None):
    """
    保存为 NIfTI 文件
    pred_numpy: [D, H, W] 的 numpy 数组
    """
    if affine is None:
        # 如果没有提供放射矩阵，使用默认单位矩阵
        affine = np.eye(4)

    # 创建 NIfTI 对象
    nii_img = nib.Nifti1Image(pred_numpy.astype(np.uint8), affine)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存
    nib.save(nii_img, save_path)


def inference_and_save(fold, ckpt_path):
    print(f"Inference Fold: {fold}")
    print(f"Loading weights from: {ckpt_path}")
    print(f"Results will be saved to: {SAVE_DIR}")

    # 1. 初始化模型
    model = Model(num_cls=4, num_modals=4)

    # 2. 加载权重
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # 处理 DataParallel 可能带来的 module. 前缀
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()

    # 3. 数据加载器
    test_data = MVClaDataset_k('../data/classification_nii_pz', 'test', fold)
    # 强制 batch_size=1
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # 4. 推理循环
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Saving Results"):
            images, mask_gt, name, size = data

            # name 是一个元组 ('subject_id',)，取第一个元素
            subject_id = name[0]

            images = images.to(device)
            # mask_gt = mask_gt.to(device) # 推理不需要 GT，除非你想保存对比

            # 遍历所有 15 种模态组合
            # 如果你只想保存全模态，把 range(15) 改为 range(1)
            for n in range(15):
                mask_input = get_val_mask(n, 1).to(device)

                # 模型推理
                output = model(images, mask_input)

                # 获取预测类别 [1, 4, D, H, W] -> [1, D, H, W]
                # argmax 会返回 0, 1, 2, 3
                pred_idx = torch.argmax(output, dim=1)

                # 转为 Numpy [D, H, W]
                pred_np = pred_idx.squeeze(0).cpu().numpy().astype(np.uint8)

                # 定义保存路径
                # 结构: ./inference_results/00_Full_1111/Subject001.nii.gz
                sub_folder = get_case_name(n)
                save_filename = f"{subject_id}.nii.gz"
                save_path = os.path.join(SAVE_DIR, sub_folder, save_filename)

                # 保存
                # 注意：这里 affine 使用了默认值。
                # 如果你的 Dataset 返回了 affine 信息，请在这里传入，否则图像的空间位置可能会丢失。
                save_nifti(pred_np, save_path)

    print("\nInference and saving completed!")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    fold = 0
    # 请替换为你的实际权重路径
    ckpt_path = '/raid/zhouyongsong/Eye_neoplasms/fuse/pt_mmformer/0/155full_dice_0.8494_missing_dice_0.7809.pth'

    inference_and_save(fold, ckpt_path)
