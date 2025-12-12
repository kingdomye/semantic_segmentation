"""
Dataset VOC2012 description
JPEGImages: Original RGB Images (Input X)
SegmentationClass: Ground Truth Segmentation Masks (Output Y)
"""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


class VOCSegmentDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClass')

        split_file_path = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        with open(split_file_path, 'r+') as f:
            self.file_names = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        file_name = self.file_names[item]

        img_path = os.path.join(self.image_dir, f'{file_name}.jpg')
        label_path = os.path.join(self.label_dir, f'{file_name}.png')

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        image = tv_tensors.Image(image)
        label = tv_tensors.Mask(label)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label


def get_transform(train=True):
    transform = []
    if train:
        transform.append(v2.RandomResize(min_size=256, max_size=512))
        transform.append(v2.RandomCrop(256))
        transform.append(v2.RandomHorizontalFlip(p=0.5))

    transform.append(v2.ToDtype(
        dtype={
            tv_tensors.Image: torch.float32,
            tv_tensors.Mask: torch.int64,
            "others": None
        },
        scale=True
    ))
    transform.append(v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))

    return v2.Compose(transform)


if __name__ == '__main__':
    root = '../../datasets/VOC2012'
    img_transform = get_transform(train=True)

    ds = VOCSegmentDataset(root, transform=img_transform)

    # 可视化一个样本在图中
    import matplotlib.pyplot as plt
    my_image, my_label = ds[10]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_vis = my_image * std + mean
    img_vis = img_vis.clamp(0, 1)
    img_vis = img_vis.permute(1, 2, 0)

    label_vis = my_label.squeeze()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img_vis)
    plt.subplot(1, 2, 2)
    plt.title("Label")
    plt.imshow(label_vis)
    plt.show()
