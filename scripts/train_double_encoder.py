import numpy as np
import torch.nn.functional as F
import torch
import sys
from tqdm import tqdm
import torch.optim as optim
import os
import loguru
import argparse
import torch.nn as nn
import cv2 as cv
import wandb
import pickle
sys.path.append(os.getcwd())

from utils import log
from dataloader import build_dataloader
from lib import loss_factory
from config import cfg
from utils import eval
from utils import post_improve_3d
import torch.multiprocessing as mp

from lib.encoder_pretrain.ppl import mono3d_encoder
from lib.encoder_train import encoder_trainable
from lib.encoder_train import encoder_frozen
from torch.optim.lr_scheduler import CosineAnnealingLR


def load_pretrained_encoder(pretrained_path, layer):
    with torch.no_grad():
        pretrained_encoder = encoder_frozen.ResnetEncoder(num_layers=layer)
        pretrained_encoder.load_state_dict(torch.load(pretrained_path), strict=False)
        print(f"Loading pretrained weights from {pretrained_path}")

    for param in pretrained_encoder.parameters():
        param.requires_grad = False

    return pretrained_encoder

def train(cfg):
    exp_name = cfg.EXP_NAME
    layer = cfg.NET_LAYER
    restore_epoch = cfg.RESTORE_EPOCH
    lr = cfg.TRAIN.LR
    epochs = cfg.TRAIN.EPOCH
    dim_prior = cfg.DATA.DIM_PRIOR
    gt_dir = cfg.VAL.GT_DIR
    pretrained_path = cfg.PRETRAIN_PATH

    log.prepare_dirs(cfg)
    log.init_logger(cfg)
    logger = loguru.logger

    model = encoder_trainable.ResnetEncoder(num_layers=layer)
    pretrained_encoder = load_pretrained_encoder(pretrained_path, layer)

    model.cuda()
    pretrained_encoder.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    start_epoch = 0
    if restore_epoch:
        restore_path_pkl = os.path.join(cfg.CHECKPOINTS_DIR, exp_name+'_'+str(restore_epoch)+'.pkl')
        model.load_state_dict(torch.load(restore_path_pkl), strict=False)
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
    InferImgLoader_RoI = build_dataloader.build_infer_loader(cfg)

    logger.info('Start training')

    for epoch_idx in range(start_epoch, epochs):

        model.train()
        # scheduler.step()
        for batch_idx, sample in enumerate(TrainImgLoader_RoI):
            global_step = 1 + batch_idx + len(TrainImgLoader_RoI) * epoch_idx

            batch_input = build_dataloader.process_batch_data(sample)

            pretrained_encoder_features = pretrained_encoder(batch_input['l_img'])

            # location_xy, location_z, orientation_conf
            result = model(batch_input['l_img'], batch_input['bbox2d'], pretrained_encoder_features, train=True)
            pred_3D = result[:3]
            loss_mse = result[-1]

            loss_3d = loss_factory.build_loss(pred_3D,
                                         batch_input['batch_RoI_points'],
                                         batch_input['batch_lidar_y_center'],
                                         batch_input['batch_lidar_orient'],
                                         batch_input['batch_lidar_density'],
                                         batch_input['P2'],
                                         batch_input['bbox2d'],
                                         batch_input['batch_dim']
                                         )

            if loss_3d is None or loss_mse is None:
                print('no valid loss at: ', global_step, pred_3D[0][:, 1])
                continue

            loss = loss_mse*5 + loss_3d

            optimizer.zero_grad()
            loss.backward()
            nan_flag = 0

            for name, parms in model.named_parameters():
                if name in ['module.location_z.4.weight', 'module.location_z.4.bias',
                            'location_z.4.weight', 'location_z.4.bias']:
                    # 检查参数的梯度中是否存在 NaN 值
                    if torch.sum(parms.grad != parms.grad) > 0:
                        logger.warning('loss back NAN, ignore! continue training')
                        nan_flag = 1

            if not nan_flag:
                optimizer.step()

            viz_dict['vis_cls'] += float(loss) # 累积当前可视化过程中的分类损失值
            viz_dict['epoch_cls'] += float(loss) # 累积整个训练周期（epoch）中的分类损失值
            viz_dict['vis_count'] += 1  # 记录累积的损失值的次数
            if viz_dict['vis_count'] % 50 == 0 and viz_dict['vis_count'] > 0:
                logger.info(
                    "Epoch_idx: {}, global_step: {}, loss: {:.4f}, max: {} epochs".format(
                        epoch_idx, global_step, float(viz_dict['vis_cls'] / 100), cfg.TRAIN.EPOCH
                    )
                )
                viz_dict['vis_cls'] = 0

        logger.info("Epoch: {}; Average loss: {}".format(epoch_idx,
                                                      viz_dict['epoch_cls'] /len(TrainImgLoader_RoI)))

        checkpoints_path = os.path.join(cfg.CHECKPOINTS_DIR, '{}_{}.pkl'.format(exp_name, epoch_idx))
        optim_path = os.path.join(cfg.CHECKPOINTS_DIR, '{}_optim_{}.pkl'.format(exp_name, epoch_idx))
        logger.info(
            "Saving checkpoint at {}. Epoch: {}, Global_step: {}".format(
                checkpoints_path, epoch_idx, global_step
            )
        )
        torch.save(model.state_dict(), checkpoints_path)
        torch.save(optimizer.state_dict(), optim_path)


        ###########################################################################
        # Evaluation
        ###########################################################################
        save_dir_exp = os.path.join(cfg.INFER.SAVE_DIR,
                                    os.path.splitext(os.path.basename(checkpoints_path))[0] + '/data')
        metrics = eval_one_epoch(save_dir_exp, InferImgLoader_RoI, model, dim_prior, gt_dir, epoch_idx, pretrained_encoder,ap_mode=40)

        viz_dict['epoch_cls'] = 0

def eval_one_epoch(save_dir_exp, InferImgLoader_RoI, model, dim_prior, gt_dir, epoch_idx, pretrained_encoder, ap_mode=40):
    if not os.path.exists(save_dir_exp):
        os.makedirs(save_dir_exp)

    with torch.no_grad():
        model.eval()
        for batch_idx, sample in tqdm(enumerate(InferImgLoader_RoI)):
            batch_input = build_dataloader.process_batch_data(sample)
            P2 = batch_input['P2'][0].cpu().numpy()
            bbox2d = batch_input['bbox2d'][0].cpu().numpy()
            det_2D = batch_input['det_2D'][0].cpu().numpy()
            file_name = batch_input['file_name'][0]

            if bbox2d.shape[0] < 1:
                np.savetxt('{}/{}.txt'.format(save_dir_exp, file_name), np.array([]), fmt='%s')
                continue

            pretrained_encoder_features = pretrained_encoder(batch_input['l_img'])

            result = model(batch_input['l_img'], batch_input['bbox2d'], pretrained_encoder_features, train=False)
            p_locxy, p_locZ, p_ortConf = result[:3]

            p_locXYZ = torch.cat([p_locxy, p_locZ], dim=1)


            fx, fy, cx, cy = P2[0][0], P2[1][1], P2[0][2], P2[1][2]

            det_3D = np.zeros((p_locXYZ.shape[0], 16), dtype=object)
            det_3D[:, 0] = ['Car' for _ in range(p_locXYZ.shape[0])]
            det_3D[:, 4:8] = det_2D[:, 1:5]
            det_3D[:, -1] = det_2D[:, -1]
            '''car dimension'''
            det_3D[:, 8:11] = [np.array(dim_prior[2]) for _ in range(p_locXYZ.shape[0])] # 把 car 的size先验(长,宽,高) 填充到det_3d[8:11]

            for i in range(len(p_locXYZ)):
                p, b = p_locXYZ[i], det_2D[i, 1:5]
                h, w, center_x, center_y = b[3] - b[1], b[2] - b[0], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                proj_box_center = ((F.sigmoid(p[:2]) - 0.5) * torch.tensor([w, h]).cuda() + \
                                   torch.tensor([center_x, center_y]).cuda() - \
                                   torch.tensor([cx, cy]).cuda()) / torch.tensor([fx, fy]).cuda()
                proj_box_center = torch.cat([proj_box_center, torch.tensor([1.]).cuda()]) # [x_proj, y_proj, 1]
                location_3d = p[2] * proj_box_center
                det_3D[i, 11:14] = location_3d.cpu().numpy()

                alpha_ratio = F.normalize((p_ortConf[i].unsqueeze(0))).squeeze(0)
                estimated_theta = torch.atan2(alpha_ratio[0], alpha_ratio[1])
                det_3D[i, 3] = float(estimated_theta)

                det_3D[i, 12] += float(det_3D[i, 8]) / 2
                det_3D[i, -2] = det_3D[i, 3] + np.arctan2(det_3D[i, 11], det_3D[i, 13])

            det_3D[:, 1:] = np.around(det_3D[:, 1:].astype(np.float64), decimals=5)
            np.savetxt('{}/{}.txt'.format(save_dir_exp, file_name), det_3D, fmt='%s')

        post_improve_3d.post_3d(save_dir_exp, save_dir_exp)
        eval_results_ap_11 = eval.eval_from_scrach(gt_dir, save_dir_exp, ap_mode=11)
        eval_results_ap_40 = eval.eval_from_scrach(gt_dir, save_dir_exp, ap_mode=40)

        metrics = {
            'ap_11_BEV_0.7_Easy': eval_results_ap_11.get("bev@0.70")[0],
            'ap_11_BEV_0.7_Mod': eval_results_ap_11.get("bev@0.70")[1],
            'ap_11_BEV_0.7_Hard': eval_results_ap_11.get("bev@0.70")[2],

            'ap_11_3D_0.7_Easy': eval_results_ap_11.get("3d@0.70")[0],
            'ap_11_3D_0.7_Mod': eval_results_ap_11.get("3d@0.70")[1],
            'ap_11_3D_0.7_Hard': eval_results_ap_11.get("3d@0.70")[2],

            'ap_40_BEV_0.5_Easy': eval_results_ap_40.get("bev@0.50")[0],
            'ap_40_BEV_0.5_Mod': eval_results_ap_40.get("bev@0.50")[1],
            'ap_40_BEV_0.5_Hard': eval_results_ap_40.get("bev@0.50")[2],

            'ap_40_3D_0.5_Easy': eval_results_ap_40.get("3d@0.50")[0],
            'ap_40_3D_0.5_Mod': eval_results_ap_40.get("3d@0.50")[1],
            'ap_40_3D_0.5_Hard': eval_results_ap_40.get("3d@0.50")[2],
        }

        return metrics

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