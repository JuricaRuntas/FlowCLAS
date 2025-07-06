from torchvision.datasets import CocoDetection
from typing import List
from pathlib import Path

class FilteredCocoDataset(CocoDetection):
    """A filtered version of the COCO dataset that excludes specified classes."""
    
    def __init__(self, root: Path, json_annotation_file: Path, exclude_classes: List[str], min_size: int = 480):
        """
        Args:
            root (Path): Path to the directory containing images.
            json_annotation_file (Path): Path to the COCO JSON annotation file.
            exclude_classes (List[str]): List of class names to exclude from the dataset.
            min_size (int): Minimum size for images to be included in the dataset.
        """
        super().__init__(root, json_annotation_file)
        self.exclude_classes = exclude_classes
        
        exclude_category_ids = self.coco.getCatIds(catNms=exclude_classes)
        do_not_exclude_img_ids = []
        
        for image_id in self.coco.getImgIds():
            annotation_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(annotation_ids)
            image_info = self.coco.loadImgs(image_id)[0]
            
            if not any(ann["category_id"] in exclude_category_ids for ann in annotations) and \
                (image_info['height'] >= min_size and image_info['width'] >= min_size):
                
                do_not_exclude_img_ids.append(image_id)
        
        self.ids = do_not_exclude_img_ids


class CocoDatasetWithoutCityscapesClasses(FilteredCocoDataset):
    """A filtered version of the COCO dataset that excludes classes visually overlapping with Cityscapes classes."""
    
    def __init__(self, root: Path, json_annotation_file: Path, min_size: int = 480, ood_id: int = 254, id_id: int = 0):
        """
        Args:
            root (Path): Path to the directory containing COCO images.
            json_annotation_file (Path): Path to the COCO JSON annotation file.
            min_size (int): Minimum size for images to be included in the dataset.
            ood_id (int): ID for out-of-distribution objects in COCO images.
            id_id (int): ID for in-distribution objects in COCO images.
        """
        
        # These classes are visually overlapping with Cityscapes classes and
        # should be excluded for (pseudo) outlier exposure
        self.exclude_classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "traffic sign"]
        self.min_size = min_size
        self.ood_id = ood_id
        self.id_id = id_id
        super().__init__(root, json_annotation_file, self.exclude_classes, min_size)
        