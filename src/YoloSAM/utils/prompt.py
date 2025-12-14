from typing import List, Tuple, Optional
import numpy as np

class Prompt:
    """Base class for generating prompts for SAM."""
    def __init__(self):
        pass
    
    def generate(self, *args, **kwargs):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate method")


class BoxPromptGenerator(Prompt):
    """Generates bounding box prompts for SAM from masks or YOLO detections."""
    def __init__(
        self,
        enable_direction_aug: bool = True,
        enable_size_aug: bool = True,
        image_shape: Tuple[int, int] = (1024, 1024),
    ):
        super().__init__()
        self.enable_direction_aug = enable_direction_aug
        self.enable_size_aug = enable_size_aug
        self.image_shape = image_shape

    def generate(self, 
                 mask: Optional[np.ndarray] = None, 
                 yolo_boxes: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate bounding boxes from mask or YOLO detections."""
        if mask is not None:
            return self.generate_from_mask(mask)
        elif yolo_boxes is not None:
            return self.generate_from_yolo(yolo_boxes, image_shape)
        else:
            raise ValueError("Either mask or yolo_boxes must be provided")

    def generate_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Generate bounding boxes with optional augmentations from mask."""
        box = self._generate_single_box_from_mask(mask)
        
        if self.enable_direction_aug:
            box = self._apply_direction_augmentation(box, mask.shape)
        if self.enable_size_aug:
            box = self._apply_size_augmentation(box, mask.shape)
        return box
    
    def generate_from_yolo(self, yolo_boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate bounding boxes from YOLO detections with optional augmentations.
        
        Args:
            yolo_boxes: YOLO format boxes [x_center, y_center, width, height] normalized (0-1)
            image_shape: (height, width) of the image
            
        Returns:
            Box in format [x_min, y_min, x_max, y_max] in pixel coordinates
        """
        # Convert YOLO format to pixel coordinates
        height, width = image_shape
        x_center, y_center, box_width, box_height = yolo_boxes
        
        # Convert from normalized to pixel coordinates
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        
        # Convert to corner format
        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2
        
        box = np.array([x_min, y_min, x_max, y_max])
        
        if self.enable_direction_aug:
            box = self._apply_direction_augmentation(box, image_shape)
        if self.enable_size_aug:
            box = self._apply_size_augmentation(box, image_shape)
        return box
    
    def _generate_single_box_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Generate a single bounding box from mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            raise ValueError("Empty mask detected - no positive pixels found")
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return np.array([x_min, y_min, x_max, y_max])
    
    def _apply_direction_augmentation(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Apply directional augmentation to the box."""
        x_min, y_min, x_max, y_max = box
        offset_x = np.random.randint(-10, 11)
        offset_y = np.random.randint(-10, 11)
        
        x_min = max(0, x_min + offset_x)
        y_min = max(0, y_min + offset_y)
        x_max = min(image_shape[1], x_max + offset_x)
        y_max = min(image_shape[0], y_max + offset_y)
        
        return np.array([x_min, y_min, x_max, y_max])
    
    def _apply_size_augmentation(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Apply size augmentation to the box."""
        x_min, y_min, x_max, y_max = box
        expand_factor_x = np.random.uniform(1, 1.3)
        expand_factor_y = np.random.uniform(1, 1.3)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        x_min = max(0, center_x - (width * expand_factor_x) / 2)
        x_max = min(image_shape[1], center_x + (width * expand_factor_x) / 2)
        y_min = max(0, center_y - (height * expand_factor_y) / 2)
        y_max = min(image_shape[0], center_y + (height * expand_factor_y) / 2)
        
        return np.array([x_min, y_min, x_max, y_max])


class PointPromptGenerator(Prompt):
    """Generates point prompts for SAM with various strategies."""
    def __init__(
        self,
        strategies: List[str] = ['positive', 'negative'],
        number_of_points: int = 3,
    ):
        super().__init__()
        self.strategies = strategies
        self.number_of_points = number_of_points
        
    def generate(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate points using multiple strategies."""
        all_points = []
        all_labels = []
        
        for strategy in self.strategies:
            points, labels = self._generate_strategy_points(mask, strategy)
            all_points.extend(points)
            all_labels.extend(labels)
            
        return np.array(all_points), np.array(all_labels)

    def _generate_strategy_points(self, mask: np.ndarray, strategy: str) -> Tuple[List[List[float]], List[int]]:
        """Generate points using a specific strategy."""
        if strategy == 'positive':
            return self._generate_positive_points(mask)
        elif strategy == 'negative':
            return self._generate_negative_points(mask)
    
    def _generate_positive_points(self, mask: np.ndarray) -> Tuple[List[List[float]], List[int]]:
        """Generate positive points inside the mask."""
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0:
            return [], []
            
        idx = np.random.choice(len(y_coords), min(self.number_of_points, len(y_coords)))
        points = [[x_coords[i], y_coords[i]] for i in idx]
        labels = [1] * len(points)
        return points, labels
    
    def _generate_negative_points(self, mask: np.ndarray) -> Tuple[List[List[float]], List[int]]:
        """Generate random points outside the mask."""
        inverse_mask = ~mask.astype(bool)
        y_coords, x_coords = np.where(inverse_mask)
        if len(y_coords) == 0:
            return [], []
            
        idx = np.random.choice(len(y_coords), min(self.number_of_points, len(y_coords)))
        points = [[x_coords[i], y_coords[i]] for i in idx]
        labels = [0] * len(points)
        return points, labels


if __name__ == "__main__":
    # Test the box generator with mask
    box_generator = BoxPromptGenerator()
    mask = np.zeros((100, 100))
    mask[50:70, 50:70] = 1
    print("box_generator.generate_from_mask(mask):", box_generator.generate_from_mask(mask))
    
    # Test the box generator with YOLO boxes
    yolo_box = np.array([0.6, 0.6, 0.4, 0.4])  # [x_center, y_center, width, height] normalized
    image_shape = (100, 100)  # (height, width)
    print("box_generator.generate_from_yolo(yolo_box, image_shape):", 
          box_generator.generate_from_yolo(yolo_box, image_shape))
    
    # Test the point generator
    point_generator = PointPromptGenerator()
    print("point_generator.generate(mask):", point_generator.generate(mask))