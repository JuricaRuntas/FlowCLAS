# FlowCLAS

## An Unofficial Implementation of [FlowCLAS: Enhancing Normalizing Flow Via Contrastive Learning For Anomaly Segmentation](https://arxiv.org/abs/2411.19888) (Lee et al.)

This repository contains minimal implementation of the proposed FlowCLAS anomaly segmentation method. The main focus is on the reproduction of results on Fishyscapes Lost & Found and Road Anomaly validation sets reported in the paper in Table 1.

Implemented normalizing flow matches the total number of trainable parameters for varying number of flow steps L, as reported in the paper in Table 6.

## Training

```
python run.py --train
```
NOTE: running this command for the first time will generate and save to disk DINOv2 ViT-L/14 features of Cityscapes+COCO dataset, if they have not been previously generated.

## Evaluation

```
python run.py --eval --weights=<PATH_TO_WEIGHTS>
```
NOTE: running this command for the first time will generate and save to disk DINOv2 ViT-L/14 features of Fishyscapes Static, Fishyscapes Lost & Found and Road Anomaly datasets, if they have not been previously generated.

## Preliminary Results

#### Fishyscapes Lost & Found
| Results | AP ↑ | AUROC ↑ | FPR@95 ↓ |
|--------|---------|---------|----------|
| Paper (Lee et al., Epoch 600) | **72.3** | - | **1.6** |
| Ours (Epoch 44) | 57.68 | 98.63 | 6.09 |

#### Road Anomaly
| Results | AP ↑ | AUROC ↑ | FPR@95 ↓ |
|--------|---------|---------|----------|
| Paper (Lee et al., Epoch 600) | **91.8** | - | **5.7** |
| Ours (Epoch 44) | 79.99 | 96.97 | 12.12 |

#### Fishyscapes Static
| Results | AP ↑ | AUROC ↑ | FPR@95 ↓ |
|--------|---------|---------|----------|
| Paper (Lee et al., Epoch 600) | - | - | - |
| Ours (Epoch 44) | 95.04 | 99.91 | 0.37 |