import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch.nn.functional as F
import torch
import sys
from tqdm import tqdm
import torch.optim as optim

import loguru
import argparse
import torch.nn as nn
import cv2 as cv
import wandb
import pickle
sys.path.append(os.getcwd())

from utils import log
from dataloader import build_dataloader
# from lib import loss_factory
from config import cfg
# from utils import eval
from utils import post_improve_3d
import torch.multiprocessing as mp

from lib.encoder_pretrain.ppl.ppl_text_encoder_multi import PPLTextEncoderMulti
from lib.encoder_pretrain.ppl.mono3d_encoder import ResnetEncoder
# from torch.optim.lr_scheduler import CosineAnnealingLR



def train(cfg):
    exp_name = cfg.EXP_NAME
    layer = cfg.NET_LAYER
    restore_epoch = cfg.RESTORE_EPOCH
    epochs = cfg.TRAIN.EPOCH

    log.prepare_dirs(cfg)
    log.init_logger(cfg)
    logger = loguru.logger

    class_names = cfg.TRAIN.KEYPOINT_LIST
    image_encoder = ResnetEncoder(num_layers=layer).to('cuda')


    # gt bbox2d finetune
    checkpoint_path = "/root/autodl-tmp/GGA/ckp/pretrain/VirPro_gt_bbox2d_finetune_25.pkl"
    state_dict = torch.load(checkpoint_path, map_location='cuda')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_k = k.replace("backbone.", "encoder.", 1)  # 只替换第一个前缀
        else:
            new_k = k
        new_state_dict[new_k] = v
    missing, unexpected = image_encoder.load_state_dict(new_state_dict, strict=False)
    print("missing: ")
    print(missing)
    print("unexpected: ")
    print(unexpected)
    print("VirPro stage 1 on raw is loaded")

    text_encoder = PPLTextEncoderMulti(class_names, 4).to('cuda')


    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    optimizer = optim.AdamW(
        [
            {"params": text_encoder.parameters(), "lr": 1e-4},
            {"params": image_encoder.parameters(), "lr": 1e-4},
                ],
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.2,
    )
    start_epoch = 0
    if restore_epoch: # 0 代表 false，非零代表 true
        restore_path_pkl = os.path.join(cfg.CHECKPOINTS_DIR, exp_name+'_'+str(restore_epoch)+'.pkl')
        image_encoder.load_state_dict(torch.load(restore_path_pkl), strict=False)
        opt_restore_path_pkl = os.path.join(cfg.CHECKPOINTS_DIR, exp_name+'_optim_'+str(restore_epoch)+'.pkl')
        optimizer.load_state_dict(torch.load(opt_restore_path_pkl))
        start_epoch = restore_epoch

    viz_dict = {
        'vis_cls': 0,
        'vis_count': 0,
        'epoch_cls': 0,
    }

    global_step = 0

    TrainImgLoader_RoI = build_dataloader.build_train_loader(cfg)


    for epoch_idx in range(start_epoch, epochs):

        image_encoder.train()
        # scheduler.step()
        for batch_idx, sample in enumerate(TrainImgLoader_RoI):

            global_step = 1 + batch_idx + len(TrainImgLoader_RoI) * epoch_idx
            batch_input = build_dataloader.process_batch_data(sample)

            # image modality
            img_feat, img_roi_feat = image_encoder(batch_input['l_img'], batch_input['bbox2d'])
            conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1) # Resnet101: 1024->512, Resnet34:256->512
            img_feat = conv(img_feat) # [16,512,24,77]

            # text modality
            cls_token = text_encoder.get_cls_token(img_feat)
            text_embed, loss_prompt = text_encoder(cls_token, img_feat) # text_embed: [16, 3, 512]

            # contrastive matching
            loss_contrastive = image_encoder.roi_contrastive_matching(img_roi_feat, text_embed)
            loss = loss_contrastive + loss_prompt*4

            optimizer.zero_grad()
            loss.backward()

            nan_flag = 0
            for name, param in image_encoder.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Warning: NaN or Inf detected in gradients of {name}")
                        nan_flag = 1
            if nan_flag == 0:
                optimizer.step()


            viz_dict['vis_cls'] += float(loss)
            viz_dict['epoch_cls'] += float(loss)
            viz_dict['vis_count'] += 1

            if viz_dict['vis_count'] % 50 == 0 and viz_dict['vis_count'] > 0:
                logger.info(
                    "Epoch_idx: {}, global_step: {}, loss: {:.4f}, max: {} epochs".format(
                        epoch_idx, global_step, float(viz_dict['vis_cls'] / 50), cfg.TRAIN.EPOCH
                    )
                )
                viz_dict['vis_cls'] = 0

        viz_dict['epoch_cls'] = 0

        checkpoints_path = os.path.join(cfg.CHECKPOINTS_DIR, '{}_{}.pkl'.format(exp_name, epoch_idx))
        optim_path = os.path.join(cfg.CHECKPOINTS_DIR, '{}_optim_{}.pkl'.format(exp_name, epoch_idx))
        logger.info(
            "Saving checkpoint at {}. Epoch: {}, Global_step: {}".format(
                checkpoints_path, epoch_idx, global_step
            )
        )
        torch.save(image_encoder.state_dict(), checkpoints_path)
        torch.save(optimizer.state_dict(), optim_path)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="PyTorch Training model")
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

    train(cfg)
