from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
_C = CN(new_allowed=True)
_C.SYSTEM = CN(new_allowed=True)
_C.DATASETS = CN(new_allowed=True)
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# System configuration
_C.SYSTEM.NUM_CPUS = 8
_C.SYSTEM.SEED = 1337
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# COCO
_C.DATASETS.COCO = CN(new_allowed=True)
_C.DATASETS.COCO.ROOT = "/hdd/datasets/coco/train2017"
_C.DATASETS.COCO.ANNOTATIONS = "/hdd/datasets/coco/annotations/instances_train2017.json"
_C.DATASETS.COCO.OOD_ID = 254 # used when creating segmentation masks for OOD objects in COCO images
_C.DATASETS.COCO.ID_ID = 0
_C.DATASETS.COCO.MIN_SIZE = 480 # only consider images with H >= MIN_SIZE AND W >= MIN_SIZE

# OOD instances will be randomly rescaled before pasting on Cityscapes images
_C.DATASETS.COCO.RANDOM_RESCALE_LOWER = 96
_C.DATASETS.COCO.RANDOM_RESCALE_UPPER = 512
_C.DATASETS.COCO.MAX_OOD_INSTANCES_TO_PASTE = 3 # per Cityscapes image for given COCO image

# Cityscapes
_C.DATASETS.CITYSCAPES = CN(new_allowed=True)
_C.DATASETS.CITYSCAPES.ROOT = "/hdd/datasets/cityscapes"
_C.DATASETS.CITYSCAPES.SPLIT = "train"
_C.DATASETS.CITYSCAPES.MODE = "fine"
_C.DATASETS.CITYSCAPES.TARGET_TYPE = "semantic"

# RoadAnomaly
_C.DATASETS.ROAD_ANOMALY = CN(new_allowed=True)
_C.DATASETS.ROAD_ANOMALY.ROOT = "/hdd/datasets/road_anomaly"

# Fishyscapes Lost&Found
_C.DATASETS.FISHYSCAPES_LAF = CN(new_allowed=True)
_C.DATASETS.FISHYSCAPES_LAF.ROOT = "/hdd/datasets/fs_lost_and_found"

# Fishyscapes Static
_C.DATASETS.FISHYSCAPES_STATIC = CN(new_allowed=True)
_C.DATASETS.FISHYSCAPES_STATIC.ROOT = "/hdd/datasets/fs_static_val"

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Feature extractor
_C.BACKBONE = CN(new_allowed=True)
_C.BACKBONE.ARCHITECTURE = "dinov2_vitl14"
_C.BACKBONE.PATCH_SIZE = 14
_C.BACKBONE.EMBED_DIM = 1024
_C.BACKBONE.NORM_MEAN = [0.485, 0.456, 0.406]
_C.BACKBONE.NORM_STD = [0.229, 0.224, 0.225]
# ---------------------------------------------------------------------------- #



def get_cfg_defaults():
    return _C.clone()