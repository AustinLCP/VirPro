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

_C.RESTORE_PATH = "Pretrain_ResNet101.pkl"
_C.RESTORE_EPOCH = 0

_C.LOG_DIR = './log'
_C.CHECKPOINTS_DIR = 'ckp/pretrain'
_C.PRETRAIN_PATH = None

_C.TRAIN.KEYPOINT_LIST = ['Car']
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.EPOCH = 50
_C.TRAIN.LR = 1e-4

_C.TRAIN.WEIGHT_FILE = 'data/kitti/data_file/kitti_raw_training_weight.txt'

_C.TRAIN.TRAIN_FILE = 'data/kitti/data_file/split/train_raw.txt'
# _C.TRAIN.TRAIN_FILE = 'data/kitti/data_file/split/train_3d.txt'


_C.TRAIN.IMAGE_HW = (370, 1232)
_C.TRAIN.SAMPLE_ROI_POINTS = 100
_C.TRAIN.SAMPLE_LOSS_POINTS = 100
_C.TRAIN.WORKS = 16
_C.TRAIN.FLIP = 0.0

_C.VAL.WORKS = 16
_C.VAL.SPLIT_FILE = 'data/kitti/data_file/split/val.txt'
# <object_type> <truncated> <occluded> <alpha> <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> <dimensions_3d> <location_3d> <rotation_y>
_C.VAL.GT_DIR = 'data/kitti/KITTI3D/training/label_2'


_C.INFER.WORKS = 16
_C.INFER.DET_2D_PATH = 'data/kitti/KITTI3D/training/rgb_detections/val/'
# _C.INFER.DET_2D_PATH = 'data/kitti/KITTI3D/training/gt_2d_bbox/'
# _C.INFER.DET_2D_PATH = 'data/kitti/KITTI3D/testing/bbox2d/'
_C.INFER.SAVE_DIR = 'pred'


_C.DATA.CLS_LIST = ['Car','Cyclist','Pedestrian']
_C.DATA.MODE = 'KITTI raw'
# _C.DATA.MODE = 'KITTI 3D'
_C.DATA.ROOT_3D_PATH = 'data/kitti/KITTI3D/training'
_C.DATA.KITTI_RAW_PATH = 'data/kitti/raw_data'


# _C.DATA.RoI_POINTS_DIR = 'weakm3d_RoI_points' # kitti 3d
_C.DATA.RoI_POINTS_DIR = 'lidar_RoI_points' # kitti raw


_C.DATA.TYPE = ['Car', 'Cyclist', 'Pedestrian']
_C.DATA.IMAGENET_STATS_MEAN = [0.485, 0.456, 0.406]
_C.DATA.IMAGENET_STATS_STD = [0.229, 0.224, 0.225]
_C.DATA.DIM_PRIOR = [[0.8, 1.8, 0.8], [0.6, 1.8, 1.8], [1.6, 1.8, 4.]]


