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
        valid_img_ids = []
        
        for image_id in self.coco.getImgIds():
            annotation_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(annotation_ids)
            image_info = self.coco.loadImgs(image_id)[0]
            
            filtered_annotations = [ann for ann in annotations if ann['category_id'] not in exclude_category_ids]

            if len(filtered_annotations) > 0 and image_info['height'] >= min_size and image_info['width'] >= min_size:
                valid_img_ids.append(image_id)

        self.ids = valid_img_ids
        
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        filtered_target = [ann for ann in target if ann['category_id'] not in self.coco.getCatIds(catNms=self.exclude_classes)]
        return img, filtered_target
    

class CocoDatasetWithoutCityscapesClasses(FilteredCocoDataset):
    """A filtered version of the COCO dataset that excludes classes visually overlapping with Cityscapes classes."""
    
    def __init__(self, root: Path, json_annotation_file: Path, min_size: int = 480):
        """
        Args:
            root (Path): Path to the directory containing COCO images.
            json_annotation_file (Path): Path to the COCO JSON annotation file.
            min_size (int): Minimum size for images to be included in the dataset.
        """
        
        # These classes are visually overlapping with Cityscapes classes and
        # should be excluded for (pseudo) outlier exposure
        self.exclude_classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "traffic sign", "train", "rider", "stop sign"]
        self.min_size = min_size
        super().__init__(root, json_annotation_file, self.exclude_classes, min_size)
        