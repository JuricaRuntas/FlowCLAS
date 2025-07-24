from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
_C = CN(new_allowed=True)
_C.SYSTEM = CN(new_allowed=True)
_C.DATASETS = CN(new_allowed=True)
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# System configuration
_C.SYSTEM.NUM_WORKERS = 16
_C.SYSTEM.SEED = 666
_C.SYSTEM.DINOV2_FEATURES_ROOT = "/workspace/dinov2_features"
_C.SYSTEM.OUTPUT_DIR = "./flowclas_output_dir"
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
_C.DATASETS.IGNORE_LABEL = 255

# COCO
_C.DATASETS.COCO = CN(new_allowed=True)
_C.DATASETS.COCO.ROOT = "/workspace/datasets/coco/train2017"
_C.DATASETS.COCO.ANNOTATIONS = "/workspace/datasets/coco/annotations/instances_train2017.json"
_C.DATASETS.COCO.MIN_SIZE = 480 # only consider images with H >= MIN_SIZE AND W >= MIN_SIZE

# OOD instances will be randomly rescaled before pasting on Cityscapes images
_C.DATASETS.COCO.RANDOM_RESCALE_LOWER = 96
_C.DATASETS.COCO.RANDOM_RESCALE_UPPER = 512

# Cityscapes
_C.DATASETS.CITYSCAPES = CN(new_allowed=True)
_C.DATASETS.CITYSCAPES.ROOT = "/workspace/datasets/cityscapes"
_C.DATASETS.CITYSCAPES.SPLIT = "train"
_C.DATASETS.CITYSCAPES.MODE = "fine"
_C.DATASETS.CITYSCAPES.TARGET_TYPE = "semantic"

# RoadAnomaly
_C.DATASETS.ROAD_ANOMALY = CN(new_allowed=True)
_C.DATASETS.ROAD_ANOMALY.ROOT = "/workspace/datasets/road_anomaly"

# Fishyscapes Lost&Found
_C.DATASETS.FISHYSCAPES_LAF = CN(new_allowed=True)
_C.DATASETS.FISHYSCAPES_LAF.ROOT = "/workspace/datasets/fs_lost_and_found"

# Fishyscapes Static
_C.DATASETS.FISHYSCAPES_STATIC = CN(new_allowed=True)
_C.DATASETS.FISHYSCAPES_STATIC.ROOT = "/workspace/datasets/fs_static_val"

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


# ---------------------------------------------------------------------------- #
# Normalizing flow
_C.NORMALIZING_FLOW = CN(new_allowed=True)
_C.NORMALIZING_FLOW.NUM_STEPS = 8
_C.NORMALIZING_FLOW.PROJECTION_HEAD_DIM = 256
_C.NORMALIZING_FLOW.NUM_FEATURES = _C.BACKBONE.EMBED_DIM
_C.NORMALIZING_FLOW.TEMPERATURE = 0.1
_C.NORMALIZING_FLOW.NUM_EPOCHS = 600
_C.NORMALIZING_FLOW.BATCH_SIZE = 32
_C.NORMALIZING_FLOW.LR = 1e-5
_C.NORMALIZING_FLOW.WEIGHT_DECAY = 1e-5
_C.NORMALIZING_FLOW.WARMUP_EPOCHS = 10
_C.NORMALIZING_FLOW.ALPHA = 0.07 # loss hyperparameter
_C.NORMALIZING_FLOW.MAX_NORM = 5.0
_C.NORMALIZING_FLOW.FEATURE_NOISE_STD = 0.01
# ---------------------------------------------------------------------------- #


def get_cfg_defaults():
    return _C.clone()