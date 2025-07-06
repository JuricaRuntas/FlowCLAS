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
# COCO dataset
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

# Cityscapes dataset
_C.DATASETS.CITYSCAPES = CN(new_allowed=True)
_C.DATASETS.CITYSCAPES.ROOT = "/hdd/datasets/cityscapes"
_C.DATASETS.CITYSCAPES.SPLIT = "train"
_C.DATASETS.CITYSCAPES.MODE = "fine"
_C.DATASETS.CITYSCAPES.TARGET_TYPE = "semantic"

# ---------------------------------------------------------------------------- #


def get_cfg_defaults():
    return _C.clone()