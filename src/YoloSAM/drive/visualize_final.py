import sys
import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# 导入你项目中的模块
from src.YoloSAM.models.sam import SAMModel
from src.YoloSAM.utils.config import SAMFinetuneConfig, SAMDatasetConfig  # 借用Config来初始化模型


# =========================================================
# 1. 定义一个专业的 Inference 类
# =========================================================
class YoloSAMInference:
    def __init__(self, yolo_path, sam_path, device='mps'):
        self.device = torch.device(device)

        # --- 1. 加载 YOLO ---
        print(f"Loading YOLO from: {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        print("✅ YOLO Model Loaded.")

        # --- 2. 加载你微调的 SAMModel ---
        print(f"Loading Fine-tuned SAM from: {sam_path}")
        self.sam_model = self._load_finetuned_sam(sam_path)
        self.sam_model.to(self.device)
        self.sam_model.eval()  # 切换到评估模式
        print("✅ Fine-tuned SAM Model Loaded.")

    def _load_finetuned_sam(self, checkpoint_path):
        # 使用与训练时相同的配置来初始化模型骨架
        # 注意：这里的 sam_path 是为了初始化 SAMModel 类，它不会被实际加载
        config = SAMFinetuneConfig(model_type='vit_b', sam_path='./runs/sam_vit_b_01ec64.pth', device='mps')
        model = SAMModel(config)

        # 加载你训练好的权重字典
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # 🔥 核心修复：如果权重被包裹在 'model_state_dict' 里，先把它取出来
        if 'model_state_dict' in state_dict:
            print("📦 Checkpoint format detected, extracting 'model_state_dict'...")
            state_dict = state_dict['model_state_dict']

        # 加载权重到模型骨架
        model.load_state_dict(state_dict)
        return model

    def predict(self, image_path, yolo_conf=0.01, yolo_iou=0.5, image_size=1024):
        # --- 1. 图像预处理 ---
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        # 模拟 Dataset 中的 Resize
        original_shape = image_np.shape[:2]
        resized_image_np = cv2.resize(image_np, (image_size, image_size))

        # --- 2. YOLO 推理 ---
        yolo_results = self.yolo_model.predict(resized_image_np, conf=yolo_conf, iou=yolo_iou, verbose=False)

        detected_boxes = []
        if yolo_results and len(yolo_results[0].boxes) > 0:
            detected_boxes = yolo_results[0].boxes.xyxy.cpu()  # 获取所有检测框

        # --- 3. SAM 推理 (对每个检测框) ---
        # 图像需要转换为 Tensor: (C, H, W) -> (1, C, H, W)
        image_tensor = torch.from_numpy(resized_image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device).unsqueeze(0)

        all_masks = []
        if len(detected_boxes) > 0:
            with torch.no_grad():
                for box in detected_boxes:
                    prompt_box = box.unsqueeze(0).to(self.device)  # (1, 4)

                    # 使用与训练时完全相同的 forward 方法
                    pred_mask_logits, _ = self.sam_model.forward_one_image(
                        image=image_tensor,
                        bounding_box=prompt_box,
                        is_train=False
                    )

                    pred_mask_prob = torch.sigmoid(pred_mask_logits)
                    pred_mask_binary = (pred_mask_prob > 0.5).squeeze().cpu().numpy()
                    all_masks.append(pred_mask_binary)

        return {
            "original_image": resized_image_np,
            "detected_boxes": detected_boxes.numpy() if len(detected_boxes) > 0 else [],
            "predicted_masks": all_masks
        }

    # ++++++++++++++++++++ 这是正确的新版本，请使用它 ++++++++++++++++++++
    def visualize_results(self, results):
        image = results['original_image']
        boxes = results['detected_boxes']
        masks = results['predicted_masks']

        # 创建一个副本用于绘制，避免修改原始数据
        vis_image = image.copy()

        if masks:
            # 将所有 mask 合并成一个单一的布尔掩码
            combined_mask = np.zeros_like(masks[0], dtype=bool)
            for mask in masks:
                # 确保 mask 是布尔类型
                combined_mask = np.logical_or(combined_mask, mask.astype(bool))

            # 🔥 核心修复：创建一个彩色的覆盖层
            # 定义颜色 (R, G, B)，注意 OpenCV 使用 BGR 顺序
            color_bgr = (0, 255, 0)

            # 将布尔掩码转换为 uint8 格式 (0 或 255)
            binary_mask_uint8 = combined_mask.astype(np.uint8) * 255

            # 使用 findContours 找到掩码的轮廓，绘制轮廓线比填充更清晰
            contours, _ = cv2.findContours(binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color_bgr, thickness=2)  # 绘制轮廓

            # 如果你更喜欢半透明填充效果，可以使用以下代码替换上面的轮廓绘制
            # overlay = vis_image.copy()
            # alpha = 0.5 # 透明度
            # overlay[combined_mask] = color_bgr
            # vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)

        # if len(boxes) > 0:
        #     for box in boxes:
        #         x0, y0, x1, y1 = map(int, box)
        #         # 绘制检测框 (绿色)
        #         cv2.rectangle(vis_image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # 使用 Matplotlib 显示最终结果 (注意 OpenCV 的 BGR -> RGB 转换)
        plt.figure(figsize=(12, 12))
        # plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.imshow(vis_image)
        plt.title(f"End-to-End Inference\nDetected Objects: {len(boxes)}")
        plt.axis('off')
        plt.show()


# =========================================================
# 2. 自动寻找最新的权重文件
# =========================================================
# 找 YOLO
# 注意：路径可能需要根据你的实际情况微调
latest_yolo_path = "./runs/yolo_vessel_detection3/weights/best.pt"
print(f"👉 Found latest YOLO weights: {latest_yolo_path}")

# 找 SAM
# 注意：路径是 /root/autodl-tmp/run_*
latest_sam_path = "./runs/run_25/best_model.pth"
print(f"👉 Found latest SAM weights: {latest_sam_path}")

# =========================================================
# 3. 执行推理和可视化
# =========================================================
# --- 初始化推理器 ---
pipeline = YoloSAMInference(
    yolo_path=latest_yolo_path,
    sam_path=latest_sam_path
)

# --- 选择一张图片进行预测 ---
image_path = "../../../datasets/DRIVE/val/images/01_test.tif"
print(f"\n🚀 Predicting on image: {image_path}")

# --- 执行预测 ---
results = pipeline.predict(image_path)

# --- 可视化结果 ---
print(f"📊 YOLO detected {len(results['detected_boxes'])} objects.")
print("🎨 Visualizing results...")
pipeline.visualize_results(results)