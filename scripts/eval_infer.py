import os
import torch
import argparse
import sys
sys.path.append(os.getcwd())

from dataloader import build_dataloader
from lib import network
from config import cfg
from scripts.pretrain_ppl_multi import eval_one_epoch
# from scripts.train import eval_one_epoch
from lib.double_contrastive import network_clip_cls

def evaluation(cfg):
    layer = cfg.NET_LAYER  # 要在网络中使用的层数

    restore_path = cfg.RESTORE_PATH  # 预训练的模型
    dim_prior = cfg.DATA.DIM_PRIOR  # 三维物体对象(car,cyclist,pedestrians)的维度先验(dimension priors),助于预测算法调整其输出，使之符合物理世界中的实际尺寸
    gt_dir = cfg.VAL.GT_DIR  # 包含验证集真实标签(Ground Truth)数据的目录

    # 指定评估结果的存储路径
    # 不同阈值下的AP都存储在这个文件夹下
    save_dir_exp = os.path.join(cfg.INFER.SAVE_DIR,
                                os.path.splitext(os.path.basename(restore_path))[0] + '/data')
    print('Predictions saved in : {}'.format(save_dir_exp))

    # model = network.ResnetEncoder(num_layers=layer)  # 初始化模型，并规定层数
    model = network_clip_cls.ResnetEncoder(num_layers=layer)

    chekpoint = torch.load(restore_path)
    model.load_state_dict(chekpoint)  # 加载预训练模型
    model.cuda()  # 将模型的参数和缓冲区移动到 GPU，从而启用 CUDA 加速以加快计算速度

    # 设置Pytorch的默认张量 (包含单一数据类型元素的多维矩阵,类似于 NumPy 数组)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # 构建用于数据加载器，传递配置对象
    # 设置用于在支持 CUDA 的 GPU 上利用 32 位浮点数的张量类型，从而实现 GPU 加速计算
    InferImgLoader_RoI = build_dataloader.build_infer_loader(cfg)

    # 开始评估，one_epoch: 一次完整的数据递归
    text = 'Detect a car, optionally a pedestrian or a cyclist.'
    eval_one_epoch(save_dir_exp, InferImgLoader_RoI, model, dim_prior, gt_dir, 0, text, ap_mode=40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # 初始化W&B
    # wandb.login(key='b02b38b6a4b2c6f4d3d679031e93cb0cffed0249')
    # wandb.init(project="WeakM3D-ViDAR")
    #
    evaluation(cfg)
    #
    # # 提交记录结果到W&b，结束wandb
    # wandb.finish()