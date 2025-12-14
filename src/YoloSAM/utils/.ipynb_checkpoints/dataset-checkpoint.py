import os
import random
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from tqdm import tqdm

from utils.config import SAMDatasetConfig
from utils.prompt import BoxPromptGenerator, PointPromptGenerator
from utils.z_score_norm import PercentileNormalize

class SAMDataset(torch.utils.data.Dataset):
    def __init__(self, config: Union[Dict, SAMDatasetConfig]):
        self.config = config if isinstance(config, SAMDatasetConfig) else SAMDatasetConfig(**config)
        
        # Prompt generatorss
        self.box_generator = BoxPromptGenerator(
            enable_direction_aug=self.config.enable_direction_aug,
            enable_size_aug=self.config.enable_size_aug,
            image_shape=(self.config.image_size, self.config.image_size)
        )
        self.point_generator = PointPromptGenerator(
            strategies=self.config.point_prompt_types,
            number_of_points=self.config.num_points
        )
    
        if self.config.train:
            self.train_transforms = A.Compose([
                A.RandomGamma(gamma_limit=self.config.gamma_limit, p=self.config.gamma_prob), # gamma augmentation
                A.Rotate(limit=self.config.rotate_limit, p=self.config.rotate_prob),  # Random rotation between -15 and +15 degrees
                A.RandomScale(scale_limit=self.config.scale_limit, p=self.config.scale_prob),  # Random scale by ±15%
                A.HorizontalFlip(p=self.config.horizontal_flip_prob), # Horizontal flip
                A.Resize(self.config.image_size, self.config.image_size),  # Ensure final size
                PercentileNormalize(lower_percentile=self.config.percentiles[0], upper_percentile=self.config.percentiles[1]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
            
        else:
            self.val_transforms = A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                PercentileNormalize(lower_percentile=self.config.percentiles[0], upper_percentile=self.config.percentiles[1]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        
        # load paths
        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []
        self._load_dataset()
        
        if self.config.remove_nonscar:
            self._remove_nonscar()
            
        if self.config.yolo_prompt:
            self.yolo_model = YOLO(self.config.yolo_model_path)
        
    def _load_dataset(self):
        """
        Load dataset - Robust version for DRIVE/Medical datasets
        """
        image_dir = os.path.join(self.config.dataset_path, 'images')
        mask_dir = os.path.join(self.config.dataset_path, 'label')  # 注意：DRIVE数据集里有时叫 mask 有时叫 masks
        
        # 1. 容错：有些数据集文件夹叫 'mask' 而不是 'masks'
        if not os.path.exists(mask_dir) and os.path.exists(os.path.join(self.config.dataset_path, 'mask')):
             mask_dir = os.path.join(self.config.dataset_path, 'mask')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise RuntimeError(f"Dataset directories not found: {image_dir} or {mask_dir}")
        
        # 支持的扩展名增加 .tif, .tiff, .gif
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif', '.bmp')
        
        # 获取所有 mask 文件名，方便后续查找
        all_masks = os.listdir(mask_dir)
        # 建立一个 {文件名(无后缀) : 完整文件名} 的映射
        mask_map = {os.path.splitext(f)[0]: f for f in all_masks if f.lower().endswith(valid_extensions)}

        print(f"Scanning images in {image_dir}...")
        
        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith(valid_extensions):
                continue
            
            image_path = os.path.join(image_dir, img_name)
            img_stem = os.path.splitext(img_name)[0] # 获取不带后缀的文件名，如 '21_training'
            
            # === 核心匹配逻辑 ===
            found_mask_name = None
            
            # 尝试 1: 直接匹配 (21_training -> 21_training.png)
            if img_stem in mask_map:
                found_mask_name = mask_map[img_stem]
            
            # 尝试 2: 尝试加 _mask 后缀 (21_training -> 21_training_mask.gif)
            elif f"{img_stem}_mask" in mask_map:
                found_mask_name = mask_map[f"{img_stem}_mask"]
                
            # 尝试 3: 尝试 DRIVE 数据集特有的 _manual1 后缀 (21_training -> 21_manual1.gif)
            # 注意：DRIVE 的文件名通常是 21_training.tif，对应的 mask 是 21_manual1.gif
            # 我们需要把 '_training' 去掉换成 '_manual1'
            elif "_training" in img_stem:
                manual_stem = img_stem.replace("_training", "_manual1")
                if manual_stem in mask_map:
                    found_mask_name = mask_map[manual_stem]
            elif "_test" in img_stem:
                manual_stem = img_stem.replace("_test", "_manual1")
                if manual_stem in mask_map:
                    found_mask_name = mask_map[manual_stem]
            
            # 如果找到了 Mask
            if found_mask_name:
                mask_path = os.path.join(mask_dir, found_mask_name)
                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)
            # ===================
        
        if self.config.sample_size:
            if len(self.image_paths) > self.config.sample_size:
                indices = random.sample(range(len(self.image_paths)), self.config.sample_size)
                self.image_paths = [self.image_paths[i] for i in indices]
                self.mask_paths = [self.mask_paths[i] for i in indices]
            
        print(f"✅ Successfully loaded {len(self.image_paths)} image-mask pairs.")
        if len(self.image_paths) == 0:
            print(f"❌ Still 0 images. Check if your images are in {image_dir} and are .tif/.png")
            print(f"   Example image file: {os.listdir(image_dir)[0] if os.listdir(image_dir) else 'Empty dir'}")
            print(f"   Example mask file: {all_masks[0] if all_masks else 'Empty dir'}")
        
    def _remove_nonscar(self):
        """Remove non-scar images from the dataset.
            If the mask is empty (sum of mask is less than 5), it is considered as non-scar.
        """
        removed_count = 0
        valid_indices = []
        for i, mask_path in enumerate(self.mask_paths):
            mask = Image.open(mask_path)
            if np.array(mask).sum() >= 5:
                valid_indices.append(i)
            else:
                removed_count += 1
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.mask_paths = [self.mask_paths[i] for i in valid_indices]
        
        print(f"Removed {removed_count} empty masks")
        print(f"Loaded {len(self.image_paths)} images and masks")
            
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset item.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dictionary containing:
                - image: Transformed image tensor
                - mask: Mask tensor
                - prompts: Dictionary of generated prompts
                    - prompt_{i}: Dictionary of generated prompt_{i}
                - image_name: Name of the image
        """
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
        mask = np.where(mask > 0.5, 1, 0).astype(np.float32)

        # print(f"\n===== 调试：样本 {idx} - {os.path.basename(self.image_paths[idx])} =====")
        # print(f"原始mask形状: {mask.shape}, 原始mask像素值范围: [{mask.min()}, {mask.max()}]")
        # print(f"原始mask非零像素数（血管区域）: {np.count_nonzero(mask)}")
        
        # 修正mask二值化逻辑（适配DRIVE）
        # 自适应二值化：根据mask最大值判断阈值
        if mask.max() > 1.0:  # 255尺度的mask
            mask = (mask > 127).astype(np.float32)
        else:  # 1.0尺度的mask（你的情况）
            mask = (mask > 0.5).astype(np.float32)
        # 调试打印2：二值化后mask信息
        # print(f"二值化后mask求和（血管像素数）: {mask.sum()}, 形状: {mask.shape}")

        # Retry 3 times if the mask is empty (because of the transform)
        for _ in range(5):
            transformed = self.train_transforms(image=image, mask=mask) if self.config.train else self.val_transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            mask_np = mask.numpy()
            if mask_np.sum() > 0:
                break
        
        mask = mask.unsqueeze(0)
        
        # Generate prompts - return tensors directly, not lists
        points_coords = torch.zeros(0, 2, dtype=torch.float32) 
        points_labels = torch.zeros(0, dtype=torch.float32)
        boxes = torch.zeros(1, 4, dtype=torch.float32)
        
        if self.config.yolo_prompt:
            # 1. 图像格式转换 (Tensor -> Numpy)
            if isinstance(image, torch.Tensor):
                image_for_yolo = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                image_for_yolo = image
            
            # 2. YOLO 推理
            results = self.yolo_model.predict(
                image_for_yolo, 
                conf=self.config.yolo_conf_threshold, 
                iou=self.config.yolo_iou_threshold, 
                imgsz=self.config.yolo_imgsz, 
                device='cuda',  # 使用配置中的 device
                verbose=False,
                augment=False,
                rect=False
            )
            
            # 3. 处理结果
            # 检查是否真的检测到了框
            if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                # === 情况 A: YOLO 成功检测到了 ===
                detected_boxes = results[0].boxes.xyxy.cpu().numpy()
                detected_conf = results[0].boxes.conf.cpu().numpy()
                
                # 为了防止维度报错，目前你的 Pipeline 只能处理单张图一个框
                # 所以我们取置信度最高的那个框 (Top-1)
                best_idx = np.argmax(detected_conf)
                best_box = detected_boxes[best_idx] # shape (4,)

                best_box_final = best_box 
                
                best_box_final = np.clip(best_box_final, 0, self.config.image_size)
                boxes = torch.tensor(best_box_final, dtype=torch.float32).unsqueeze(0)
                
            else:
                if self.config.train:
                    box = self.box_generator.generate(mask_np)
                    box = np.array(box)
                    box = np.clip(box, 0, self.config.image_size) # 限制在 1024 以内
                    
                    boxes = torch.tensor(box, dtype=torch.float32).unsqueeze(0)
                else:
                    boxes = torch.zeros(0, 4, dtype=torch.float32)
        
        else:
            if self.config.point_prompt:
                points, labels = self.point_generator.generate(mask_np)
                points_coords = torch.tensor(points, dtype=torch.float32)
                points_labels = torch.tensor(labels, dtype=torch.float32)

            if self.config.box_prompt:
                box = self.box_generator.generate(mask_np)
            
                box = np.array(box)
                box = np.clip(box, 0, self.config.image_size)
                
                boxes = torch.tensor(box, dtype=torch.float32)

        # 1. 如果 boxes 是一维的 [4]，把它变成二维 [1, 4]
        if boxes is not None and boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)
            
        # 2. 如果 boxes 是空的 [0]，把它变成 [0, 4] (防止空张量形状不匹配)
        if boxes is not None and boxes.numel() == 0 and boxes.dim() == 1:
             boxes = boxes.reshape(0, 4)
             
        # 同理，检查 points_coords (如果有用到的话)
        if points_coords is not None and points_coords.dim() == 1:
             points_coords = points_coords.unsqueeze(0)
             
        if points_labels is not None and points_labels.dim() == 0: # scalar
             points_labels = points_labels.unsqueeze(0)

        return {
            'image': image.float(),
            'mask': mask.float(), 
            'points_coords': points_coords,
            'points_labels': points_labels,
            'boxes': boxes,
            'image_name': os.path.basename(self.image_paths[idx])
        }
    
if __name__ == "__main__":
    # Test dataset
    config = SAMDatasetConfig(
        dataset_path='./sample_data/train',
        image_size=1024,
        point_prompt=True,
        box_prompt=True,
        num_points=3,
        train=True,
        remove_nonscar=True,
        yolo_prompt=True,
        yolo_model_path='runs/yolo_scar_detection2/weights/best.pt',
        point_prompt_types=['positive'],
        sample_size=10
    )
    dataset = SAMDataset(config)
    data = dataset[0]
    print(data['image'].shape)
    print(data['mask'].shape)
    # print(data['points_coords'].shape)
    # print(data['points_labels'].shape)
    print(data['boxes'].shape)
    print(data['boxes'])
    
