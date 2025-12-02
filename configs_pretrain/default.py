#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from yacs.config import CfgNode as CN

_C = CN()
_C.TRAIN = CN()
_C.VAL = CN()
_C.INFER = CN()
_C.DATA = CN()


_C.EXP_NAME = "default"
_C.NET_LAYER = 101

# _C.RESTORE_PATH = "Pretrained_Res34.pkl"
# _C.RESTORE_PATH = "ckp_OV_WeakM3D/train_TAI_all_double_encoder_concat_mlp_mse_2/TAI_all_double_encoder_concat_mlp_mse_2_47.pkl"
_C.RESTORE_PATH = None

_C.RESTORE_EPOCH = 0

_C.LOG_DIR = './log'
# 存储和管理训练过程中的模型检查点(checkpoints),
# 用于保存训练过程中模型的状态，以便于后续的恢复和继续训练或用于评估
_C.CHECKPOINTS_DIR = 'ckp_gga/VirPro_cyclist'
# _C.PRETRAIN_PATH = "ckp_ppl/1_ppl_n_roi_5/1_ppl_n_roi_5_20.pkl"
_C.PRETRAIN_PATH = None

_C.TRAIN.KEYPOINT_LIST = ['Cyclist']
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.EPOCH = 50
_C.TRAIN.LR = 1e-4

# 每一行表示每一个训练样本的权重，调整模型在训练过程中对不同样本的重视程度
# _C.TRAIN.WEIGHT_FILE = 'data/kitti/data_file/kitti_raw_training_weight.txt'
_C.TRAIN.WEIGHT_FILE = 'data/kitti/data_file/kitti_3d_training_weight_cyclist.txt'
# _C.TRAIN.WEIGHT_FILE = 'data/kitti/data_file/kitti_3d_sample_weight.txt'

# _C.TRAIN.TRAIN_FILE = 'data/kitti/data_file/split/train_raw.txt'
_C.TRAIN.TRAIN_FILE = 'data/kitti/data_file/split/train_3d_trainval.txt'
# _C.TRAIN.TRAIN_FILE = 'data/kitti/data_file/split/train_3d.txt'


_C.TRAIN.IMAGE_HW = (370, 1232)
_C.TRAIN.SAMPLE_ROI_POINTS = 100
_C.TRAIN.SAMPLE_LOSS_POINTS = 100
_C.TRAIN.WORKS = 16
_C.TRAIN.FLIP = 0.0

_C.VAL.WORKS = 16
# 用于验证(validation)的数据分割文件的路径
_C.VAL.SPLIT_FILE = 'data/kitti/data_file/split/val.txt'
# 包含验证集真实标签(Ground Truth)数据的目录
# <object_type> <truncated> <occluded> <alpha> <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> <dimensions_3d> <location_3d> <rotation_y>
_C.VAL.GT_DIR = 'data/kitti/KITTI3D/training/label_2'


_C.INFER.WORKS = 16
# 在推理(inference)阶段所使用的二维检测结果的存储路径,推理: 使用训练好的模型对新数据进行预测的过程
_C.INFER.DET_2D_PATH = 'data/kitti/KITTI3D/training/rgb_detections/val/'
# _C.INFER.DET_2D_PATH = 'data/kitti/KITTI3D/training/gt_2d_bbox/'
# _C.INFER.DET_2D_PATH = 'data/kitti/KITTI3D/testing/bbox2d/'
_C.INFER.SAVE_DIR = 'pred'


_C.DATA.CLS_LIST = ['Car','Cyclist','Pedestrian']
# _C.DATA.MODE = 'KITTI raw'
_C.DATA.MODE = 'KITTI 3D'
_C.DATA.ROOT_3D_PATH = 'data/kitti/KITTI3D/training'
# _C.DATA.ROOT_3D_PATH = 'data/kitti/KITTI3D/testing'
_C.DATA.KITTI_RAW_PATH = 'data/kitti/raw_data'

# _C.DATA.RoI_POINTS_DIR = 'YOLO_RoI_points'
# _C.DATA.RoI_POINTS_DIR = 'weakm3d_RoI_points' # kitti 3d
_C.DATA.RoI_POINTS_DIR = 'gt_2d_bbox_cyclist' # kitti 3d gt 2d bbox
# _C.DATA.RoI_POINTS_DIR = 'lidar_RoI_points' # kitti raw



_C.DATA.TYPE = ['Car', 'Cyclist', 'Pedestrian']
# 在预处理图像数据时使用的RGB通道的均值(mean)
# 在处理图像时，每个通道的像素值通常会减去这些均值，这有助于模型更好地学习和泛化
_C.DATA.IMAGENET_STATS_MEAN = [0.485, 0.456, 0.406]
# 在预处理图像数据时使用的RGB通道的方差(standard deviation)
# 在标准化过程中，通常会使用这些标准差来将每个通道的像素值除以相应的标准差，进一步帮助网络训练的稳定性和效率
_C.DATA.IMAGENET_STATS_STD = [0.229, 0.224, 0.225]

# 三维物体对象(car,cyclist,pedestrians)的维度先验(dimension priors)
# 助于预测算法调整其输出，使之符合物理世界中的实际尺寸
_C.DATA.DIM_PRIOR = [[0.8, 1.8, 0.8], [0.6, 1.8, 1.8], [1.6, 1.8, 4.]]


