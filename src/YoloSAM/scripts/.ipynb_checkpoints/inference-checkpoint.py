import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from PIL import Image

from models.yolo import YOLOModel
from models.sam import SAMModel
from utils.config import YOLOConfig, SAMFinetuneConfig, YoloSAMInferenceConfig
from utils.prompt import BoxPromptGenerator


class YoloSAMInference:
    def __init__(
        self,
        config: YoloSAMInferenceConfig
    ):
        self.device = config.device
        
        # Initialize YOLO model
        self.yolo_config = YOLOConfig(
            checkpoint_path=config.yolo_checkpoint_path,
            device=config.device,
            conf_threshold=config.yolo_conf_threshold,
            iou_threshold=config.yolo_iou_threshold,
            max_detections=config.yolo_max_detections,
            dataset_path=None,
            val_dataset_path=None
        )
        self.yolo_model = YOLOModel(self.yolo_config)
        
        # Initialize SAM model
        self.sam_config = SAMFinetuneConfig(
            sam_path=config.sam_checkpoint_path,
            checkpoint_path=None, # Change to config.sam_checkpoint_path for training
            model_type="vit_b",
            device=config.device
        )
        self.sam_model = SAMModel(self.sam_config)
        
        # Initialize prompt generator
        self.box_prompt_generator = BoxPromptGenerator(
            enable_direction_aug=False,
            enable_size_aug=False
        )
        
        print(f"YoloSAM Inference pipeline initialized on {config.device}")
    
    def preprocess_image(self, image_path: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess image for inference."""
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Store original image
        original_image = image.copy()
        
        # Resize for SAM (1024x1024)
        sam_image = cv2.resize(image, (1024, 1024))
        sam_image_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).float() / 255.0
        sam_image_tensor = sam_image_tensor.unsqueeze(0)  # Add batch dimension
        
        return original_image, sam_image_tensor
    
    def detect_objects(self, image_path: Union[str, Path, np.ndarray]) -> List[Dict]:
        """Detect objects using YOLO model."""
        results = self.yolo_model.predict(image_path, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get box coordinates in xyxy format
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': xyxy,  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class': cls,
                        'class_name': self.yolo_config.class_names[cls] if cls < len(self.yolo_config.class_names) else f'class_{cls}'
                    })
        
        return detections
    
    def segment_with_sam(
        self, 
        sam_image_tensor: torch.Tensor, 
        detections: List[Dict],
        original_image_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Segment detected objects using SAM model."""
        results = []
        
        for detection in detections:
            bbox = detection['bbox']
            
            # Convert bbox to SAM image coordinates (1024x1024)
            h_orig, w_orig = original_image_shape[:2]
            x1, y1, x2, y2 = bbox
            
            # Scale coordinates to SAM image size (1024x1024)
            x1_sam = int(x1 * 1024 / w_orig)
            y1_sam = int(y1 * 1024 / h_orig)
            x2_sam = int(x2 * 1024 / w_orig)
            y2_sam = int(y2 * 1024 / h_orig)
            
            # Create bounding box tensor for SAM
            sam_bbox = torch.tensor([x1_sam, y1_sam, x2_sam, y2_sam], dtype=torch.float32)
            
            # Get segmentation mask from SAM
            mask, iou_pred = self.sam_model.forward_one_image(
                image=sam_image_tensor,
                bounding_box=sam_bbox,
                is_train=False
            )
            
            # Process mask
            mask = torch.sigmoid(mask)
            mask_binary = (mask > 0.5).float()
            mask_np = mask_binary[0, 0].cpu().numpy()  # Remove batch and channel dims
            
            # Scale mask back to original image size
            mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            results.append({
                'bbox': bbox,
                'mask': mask_resized,
                'confidence': detection['confidence'],
                'class': detection['class'],
                'class_name': detection['class_name'],
                'iou_prediction': float(iou_pred[0].cpu().detach().numpy())
            })
        
        return results
    
    def predict(self, image_path: Union[str, Path, np.ndarray]) -> Dict:
        """Complete inference pipeline: YOLO detection + SAM segmentation."""
        # Preprocess image
        original_image, sam_image_tensor = self.preprocess_image(image_path)
        
        # Detect objects with YOLO
        detections = self.detect_objects(image_path)
        
        if not detections:
            return {
                'image': original_image,
                'detections': [],
                'message': 'No objects detected by YOLO'
            }
        
        # Segment objects with SAM
        segmentation_results = self.segment_with_sam(
            sam_image_tensor, 
            detections, 
            original_image.shape
        )
        
        return {
            'image': original_image,
            'detections': segmentation_results,
            'message': f'Detected and segmented {len(segmentation_results)} objects'
        }
    
    def visualize_results(
        self, 
        results: Dict, 
        save_path: Optional[str] = None,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Visualize detection and segmentation results."""
        image = results['image'].copy()
        detections = results['detections']
        
        # Define colors for visualization
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for i, detection in enumerate(detections):
            color = colors[i % len(colors)]
            
            # Draw bounding box
            bbox = detection['bbox'].astype(int)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw mask overlay
            mask = detection['mask']
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0.5] = color
            image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
            
            # Add label
            label = f"{detection['class_name']}"
            if show_confidence:
                label += f" {detection['confidence']:.2f}"
            
            cv2.putText(
                image, label, 
                (bbox[0], bbox[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, color, 2
            )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Visualization saved to: {save_path}")
        
        return image
    
 
def main():
    """Example usage of the YoloSAM inference pipeline."""
    
    # Configuration
    config = YoloSAMInferenceConfig(
        yolo_checkpoint_path="runs/yolo_scar_detection2/weights/best.pt",
        sam_checkpoint_path="checkpoints/sam_vit_b_01ec64.pth",
        device="cpu"
    )
    
    # Initialize inference pipeline
    inference_pipeline = YoloSAMInference(
        config=config
    )
    
    # Single image inference
    image_path = "sample_data/val/images/Case_P004_slice_02.png"
    
    if Path(image_path).exists():
        print(f"Running inference on: {image_path}")
        
        # Predict
        results = inference_pipeline.predict(image_path)
        
        # Print results
        print(f"Results: {results['message']}")
        for i, detection in enumerate(results['detections']):
            print(f"Detection {i+1}:")
            print(f"  Class: {detection['class_name']}")
            print(f"  Confidence: {detection['confidence']:.3f}")
            print(f"  IoU Prediction: {detection['iou_prediction']:.3f}")
            print(f"  Bbox: {detection['bbox']}")
        
        # Visualize and save results
        output_image = inference_pipeline.visualize_results(
            results, 
            save_path="inference_result.jpg"
        )
        
        # Display using matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(output_image)
        plt.title("YoloSAM Inference Results")
        plt.axis('off')
        plt.show()
    
    else:
        print(f"Image not found: {image_path}")
        print("Please update the image_path in the main function")



if __name__ == "__main__":
    main()