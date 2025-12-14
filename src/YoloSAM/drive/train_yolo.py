from src.YoloSAM.utils.mask_to_yolo import MaskToYOLOConverter
from src.YoloSAM.utils.config import YOLOConfig
from src.YoloSAM.scripts.train_yolo import YOLOTrainer

converter = MaskToYOLOConverter(class_id=0)
converter.convert_dataset_inplace(
    base_path='../../../datasets/DRIVE',
    min_area=5,
    splits=['train', 'test']
)

config = YOLOConfig(
    # Model Settings
    model_type="yolo11n",
    device='mps',
    pretrained_path='../checkpoints',

    # DataSet Settings
    dataset_path='../../../datasets/DRIVE/train',
    val_dataset_path='../../../datasets/DRIVE/test',
    class_names=['scar'],

    # Training parameters
    epochs=10,
    batch_size=16,
    image_size=640,
    patience=50,

    # Augmentation (optimized for medical scars)
    mosaic=0.9,
    mixup=0.1,
    copy_paste=0.4,
    degrees=15.0,
    hsv_v=0.3,

    # Detection parameters
    iou_threshold=0.2,
    conf_threshold=0.15,
    max_detections=2,

    # Project settings
    project_name="yolo_scar_detection",
    experiment_name="enhanced_scar_detection",

    # Wandb settings
    wandb_project="YOLO-scar-detection",
    wandb_name="scar_detection_v1",
    wandb_mode="disabled"
)

trainer = YOLOTrainer(config)
results = trainer.train()
