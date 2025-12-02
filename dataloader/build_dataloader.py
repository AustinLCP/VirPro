import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from . import kitti_dataloader
from . import kitti_dataloader_stage_1


def process_batch_data(sample):
    batch_data = {}
    keys = sample.keys()
    # print(keys)
    for k in keys:
        if k == 'file_name' or k == 'img_path':
            batch_data[k] = sample[k]
        else:
            batch_data[k] = sample[k].to("cuda:0")  # 移动到 GPU 进行计算

    return batch_data

def build_train_loader(cfg):
    # train_dataset_RoI = kitti_dataloader.KITTI3D_Object_Dataset_Raw_RoI_Lidar(cfg)
    train_dataset_RoI = kitti_dataloader_stage_1.KITTI3D_Object_Dataset_Raw_RoI_Lidar(cfg)
    sampler_weights_RoI = np.loadtxt(cfg.TRAIN.WEIGHT_FILE)

    # 该对象用于根据指定的权重从数据集中对数据点进行采样
    sampler_RoI = WeightedRandomSampler(weights=sampler_weights_RoI,
                                        num_samples=len(sampler_weights_RoI),
                                        # num_samples=len(cfg.TRAIN.TRAIN_FILE),
                                        replacement=True)  # 在单个epoch内可以对一个数据点进行多次采样
    # 一个对象代表一个batch的数据内容
    TrainImgLoader_RoI = DataLoader(train_dataset_RoI,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    num_workers=cfg.TRAIN.WORKS,
                                    sampler=sampler_RoI,
                                    pin_memory=True,  # 提高 GPU 的传输速度
                                    drop_last=True)  # 如果数据集大小不能被批大小整除，则删除最后一个未完成的批处理
    return TrainImgLoader_RoI


def build_infer_loader(cfg):
    infer_dataset = kitti_dataloader.KITTI3D_Object_Dataset_BBox2D(cfg)
    InferImgLoader = DataLoader(infer_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.INFER.WORKS,
                                drop_last=False)
    return InferImgLoader


