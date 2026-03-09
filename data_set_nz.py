import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
from PIL import Image
import torchvision
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import json
import torchio as tio

# 淋巴瘤
# 黑色素瘤
# 腺样囊性癌
# 肉瘤
# 基底细胞癌
# 鳞癌
# 其他腺癌

# 囊肿
# 炎性假瘤
# 多形性腺瘤
# 脑膜瘤
# 血管瘤
# 神经鞘瘤
# 孤立性纤维瘤
# class_index = [['囊肿', '炎性假瘤', '多形性腺瘤', '脑膜瘤', '血管瘤', '神经鞘瘤', '孤立性纤维瘤'],
#                ['淋巴瘤', '黑色素瘤', '腺样囊性癌', '肉瘤', '基底细胞癌', '鳞癌', '其他腺癌']]


# class_index = [['淋巴瘤'], ['黑色素瘤'], ['腺样囊性癌'], ['肉瘤'], ['基底细胞癌'], ['鳞癌'], ['其他腺癌'], ['囊肿'], ['炎性假瘤'], ['多形性腺瘤'], ['脑膜瘤'],
#                ['血管瘤'], ['神经鞘瘤'], ['孤立性纤维瘤']]

class_index = [['预训练']]


def get_train_transform(
        rotate_deg=10,
        elastic_p=0.3, ):
    """
    返回 torchio.Compose，用于训练集。
    所有变换都会同时对 image 和 label 做一致处理。
    """
    spatial = tio.Compose([
        tio.RandomAffine(
            degrees=rotate_deg,
            isotropic=False,
            default_pad_value='minimum',
            image_interpolation='linear',
            label_interpolation='nearest'
        ),
        tio.RandomElasticDeformation(
            num_control_points=(7, 7, 7),
            max_displacement=5,
            locked_borders=2,
            p=elastic_p
        ),
    ])

    return spatial


class MVClaDataset_k(Dataset):
    def __init__(self,
                 base_dir,
                 split='train',
                 fold=0,
                 n_splits=8,
                 random_state=42,
                 weight=None,
                 test_size=0.2):  # 新增：测试集占比
        self._base_dir = base_dir
        self.fold = fold
        self.n_splits = n_splits
        self.random_state = random_state
        self.weight = weight
        self.split = split
        self.test_size = test_size
        self.indx_ = 0

        # ---------- 1. 收集所有样本 ----------
        self.class_index = class_index
        all_paths, all_labels = [], []
        for i, class_names in enumerate(self.class_index):
            for dir_name in class_names:
                class_dir = os.path.join(self._base_dir, dir_name + '_image')
                if not os.path.exists(class_dir):
                    continue
                for file in os.listdir(class_dir):
                    if file.endswith('.nii.gz') and not file.startswith('_') and os.path.exists(os.path.join(class_dir.replace('_image', '_label'), file)):
                        all_paths.append(os.path.join(class_dir, file))
                        all_labels.append(i)

        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=self.test_size,
                                     random_state=self.random_state)
        trainval_idx, test_idx = next(sss.split(all_paths, all_labels))

        if split == 'test':
            self.data = [all_paths[i] for i in test_idx]
            self.label = [all_labels[i] for i in test_idx]
            # 统计各个样本数量
            self.class_count = [0 for _ in range(len(self.class_index))]
            for l in self.label:
                self.class_count[l] += 1
            print(f'各类样本数量：{self.class_count}')
            return  # 测试集直接返回即可
        elif split == 'all':
            self.data = all_paths
            self.label = all_labels
            # 统计各个样本数量
            self.class_count = [0 for _ in range(len(self.class_index))]
            for l in self.label:
                self.class_count[l] += 1
            print(f'各类样本数量：{self.class_count}')
            return  # 全部数据集直接返回即可

        # ---------- 3. 在剩余数据上做 K 折 ----------
        trainval_paths = [all_paths[i] for i in trainval_idx]
        trainval_labels = [all_labels[i] for i in trainval_idx]

        skf = StratifiedKFold(n_splits=n_splits,
                              shuffle=True,
                              random_state=random_state)
        selected_indices = []
        for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(trainval_paths, trainval_labels)):
            if fold == fold_idx:
                if split == 'train':
                    selected_indices = train_idx
                else:
                    selected_indices = val_idx

        self.data = [trainval_paths[i] for i in selected_indices]
        self.label = [trainval_labels[i] for i in selected_indices]

        # self.data = self.data[:20]
        # self.label = self.label[:20]

        # ---------- 4. 可选：按权重过采样 ----------
        if weight is not None:
            new_data, new_labels = [], []
            for path, label in zip(self.data, self.label):
                for _ in range(weight[label]):
                    new_data.append(path)
                    new_labels.append(label)
            self.data, self.label = new_data, new_labels
        # data长度为8的倍数
        # if len(self.data) % 8 != 0:
        # 统计各类样本数量
        self.class_count = [0 for _ in range(len(self.class_index))]
        for l in self.label:
            self.class_count[l] += 1
        print(f'各类样本数量：{self.class_count}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.indx_ += 1
        # print(self.indx_)
        image_path = self.data[idx]
        label_path = self.data[idx].replace('_image', '_label')
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        # image = np.transpose(image, (3, 0, 2, 1))
        # label = np.transpose(label, (3, 0, 2, 1))

        image = normalize_3d(image)

        label = np.expand_dims(label, axis=-1)
        image = np.transpose(image, (3, 2, 1, 0))
        label = np.transpose(label, (3, 2, 1, 0))
        # image[image < 0] = 0
        # image[image > 1] = 1
        # print(image.shape)
        c, d, w, h = image.shape
        # 数据增强
        if self.split == 'train':
            # if np.random.rand() < 0.2:
            #     # 上下翻转
            #     image = image[:, ::-1, :, :]
            #     label = label[:, ::-1, :, :]
            # #     en = en[::-1, :, :]
            # if np.random.rand() < 0.2:
            #     # 左右翻转
            #     image = image[:, :, ::-1, :]
            #     label = label[:, :, ::-1, :]
            #
            if np.random.rand() < 0.5:
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]

            if np.random.rand() < 0.5:
                # 外圈裁剪,随机
                cut_size = np.random.randint(1, 16)
                image = image[:, cut_size:-cut_size, cut_size:-cut_size, cut_size:-cut_size]
                label = label[:, cut_size:-cut_size, cut_size:-cut_size, cut_size:-cut_size]
            else:
                # 外圈填充0,shape: (c, h, w, d),h和w随机填充
                padd_size = np.random.randint(1, 16)
                image = np.pad(image, ((0, 0), (padd_size, padd_size), (padd_size, padd_size), (padd_size, padd_size)), 'constant')
                label = np.pad(label, ((0, 0), (padd_size, padd_size), (padd_size, padd_size), (padd_size, padd_size)), 'constant')
            #
            if np.random.rand() < 0.5:
                # 随机平移-5到5
                dw, dh, dd = np.random.randint(-12, 12, size=3)
                # dd = 0
                image = np.roll(image, (0, dd, dw, dh), axis=(0, 1, 2, 3))
                label = np.roll(label, (0, dd, dw, dh), axis=(0, 1, 2, 3))

            if np.random.rand() < 0.5:
                # 随机伽马
                gamma = np.random.uniform(0.6, 1.5)
                image = image ** gamma

            if np.random.rand() < 0.5:
                # 随机调增亮度
                brightness = np.random.uniform(-1 / 100, 1 / 100)
                image = image + brightness

            if np.random.rand() < 0.5:
                # 添加噪声
                noise = np.random.normal(-0.005, 0.005, size=image.shape)
                image = image + noise

            # 转为tensor
            image = torch.from_numpy(image.copy()).float()
            label = torch.from_numpy(label.copy()).float()

            # if d > 24:
            #     if np.random.rand() < 0.5:
            #         # 随机裁剪
            #         cat_size = d - 24
            #         start = np.random.randint(0, (cat_size // 2) + 1)
            #         end = np.random.randint(0, (cat_size // 2) + 1)
            #         image = image[:, start:d - end, :, :]
            #         label = label[:, start:d - end, :, :]
            # if d < 16:
            #     if np.random.rand() < 0.25:
            #         # 随机填充
            #         padd_size = np.random.randint(0, 16 - d)
            #         image = torch.nn.functional.pad(image, (0, 0, padd_size, 16 - d - padd_size, 0, 0, 0, 0), 'constant')
            #         label = torch.nn.functional.pad(label, (0, 0, padd_size, 16 - d - padd_size, 0, 0, 0, 0), 'constant')
            #     elif np.random.rand() < 0.25:
            #         # 复制第一帧
            #         for i in range(16 - d):
            #             image = torch.cat((image[:, 0:1, :, :], image), dim=1)
            #             label = torch.cat((label[:, 0:1, :, :], label), dim=1)
            #     elif np.random.rand() < 0.25:
            #         # 复制最后一帧
            #         for i in range(16 - d):
            #             image = torch.cat((image, image[:, -1:, :, :]), dim=1)
            #             label = torch.cat((label, label[:, -1:, :, :]), dim=1)
            #     else:
            #         pass

            if np.random.rand() < 0.1:
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image),
                    label=tio.LabelMap(tensor=label),
                )
                transform = get_train_transform()
                transformed = transform(subject)
                image = transformed.image.data
                label = transformed.label.data


        else:
            image = torch.from_numpy(image.copy()).float()
            label = torch.from_numpy(label.copy()).float()

        label = label[0:1]
        label = F.one_hot(label.long(), num_classes=4).permute(0, 4, 1, 2, 3).squeeze(0).float()
        if self.split != 'test':
            image = F.interpolate(torch.unsqueeze(image, 0), size=(128, 128, 128), mode='trilinear', align_corners=True).squeeze(0)
            label = F.interpolate(torch.unsqueeze(label, 0), size=(128, 128, 128), mode='trilinear', align_corners=True).squeeze(0)

        # for i in range(image.shape[1]):
        #     img = preprocess(image[:, i, :, :])
        #     images.append(img)
        # image = torch.stack(images, dim=1)
        image = (image - 0.5) / 0.5
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        # # sobel提取边界
        # sobel_kernel_x = torch.tensor([[-1, 0, 1],
        #                                [-2, 0, 2],
        #                                [-1, 0, 1]], dtype=torch.float32)
        # sobel_kernel_y = torch.tensor([[-1, -2, -1],
        #                                [0, 0, 0],
        #                                [1, 2, 1]], dtype=torch.float32)
        # label_x = F.conv2d(label, sobel_kernel_x, padding=1)
        # label_y = F.conv2d(label, sobel_kernel_y, padding=1)
        # label_xy = torch.sqrt(label_x ** 2 + label_y ** 2)
        # label_xy = (label_xy - label_xy.min()) / (label_xy.max() - label_xy.min())
        # label += label_xy * 0.1

        cls = torch.tensor(self.label[idx])

        # label[2], label[3] = label[3], label[2]
        #
        # label[2][label[1] > 0.5] = 1
        # label[3][label[2] > 0.5] = 1
        if self.split == 'test':
            return image, label, self.data[idx], torch.tensor((d, w, h), dtype=torch.int32)
        else:
            return image, label, cls
        # return image, label, cls



class Resize(object):
    """
    Resize the image in a sample to a given size
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, train=True):
        self.output_size = output_size
        self.train = train

    def __call__(self, image):
        image_ = F.interpolate(image.unsqueeze(0), size=self.output_size, mode='trilinear', align_corners=True).squeeze(
            0)
        return image_


def normalize_3d(data):
    """
    将数据归一化到 0 到 1 的范围
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    data = normalized_data
    return data


if __name__ == '__main__':

    train_data = MVClaDataset_k('../data/classification_nii_pz', 'train')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    for data in train_loader:
        print(data)
