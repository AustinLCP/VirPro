# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
from mmcv.runner import load_checkpoint
import torch
import torch.distributed as dist
# from mmcv import Config, DictAction
from mmengine.config import Config, DictAction
#from mmcv.runner import get_dist_info, init_dist
from mmengine.dist import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_device, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


import torch.nn as nn
import torch.nn.functional as F

class DistillWrapper(nn.Module):
    """
    学生-教师蒸馏包装器：仅对齐 backbone 的最后一级特征做 MSE，
    同时保留原有 3D task loss（由 student.forward_train 产生）。
    """
    def __init__(self, student, teacher, kd_weight=0.25):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.kd_weight = kd_weight
        # self.kd_weight_raw = nn.Parameter(torch.tensor(float(kd_weight), dtype=torch.float32))

        # 冻结 teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # 用于暂存 student 的 backbone 输出（整个tuple/list）
        self._feat_bucket = {}

        # 在 student.backbone 上注册 hook，捕获其 forward 输出
        # 大多数 ResNet 返回 tuple/list [C2, C3, C4, C5]，我们仅用最后一级
        def _bk_hook(_m, _inp, out):
            self._feat_bucket['s_backbone_feats'] = out
        self._hook_handle = self.student.backbone.register_forward_hook(_bk_hook)


    # learnable kd_weight only
    def _parse_losses(self, losses):
        """最小实现：把各项 loss 做 mean 并累加；同时生成可记录的 log_vars（标量）。"""
        log_vars = {}
        loss = 0

        for name, value in losses.items():
            if isinstance(value, torch.Tensor):
                log_vars[name] = value.mean()
                loss = loss + log_vars[name]
            elif isinstance(value, list):
                # 兼容 list[tensor]
                log_vars[name] = sum(v.mean() for v in value)
                loss = loss + log_vars[name]
            else:
                # 跳过非 tensor 的项（如字符串、字典等）
                continue

        # runner 期望的 'loss' 键会被 OptimizerHook 用来 backward()
        log_vars['loss'] = loss

        # 转为 python 标量便于 logger 记录
        log_vars = {k: v.item() if torch.is_tensor(v) else float(v) for k, v in log_vars.items()}
        return loss, log_vars


    def train_step(self, data, optimizer):
        """
        兼容 MMCV Runner：
        - data: DataContainer 抽出来后一般是普通 Tensor/列表字典
        - 返回 dict(loss=..., log_vars=..., num_samples=...)
        """
        # 通常数据里键名和 Collect3D 对应：'img', 'gt_bboxes', 'gt_labels', ...
        # 让学生网络先做一次前向以获得任务损失，然后在 forward_train 里已加入 KD
        # 这里直接调用 wrapper 的 forward(**data) 或 forward_train(**data)
        # 为了兼容 MMDet 的日志与反传，使用学生的 _parse_losses
        losses = self.forward_train(**data)
        # BaseDetector 有 _parse_losses；我们直接复用学生的实现
        loss, log_vars = self.student._parse_losses(losses)  # type: ignore
        # loss, log_vars = self._parse_losses(losses)
        # 由 Runner/OptimHook 负责 optimizer.step()，这里只返回张量 loss
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data.get('img', None)) if hasattr(data.get('img', None), '__len__') else 1
        )
        return outputs

    @staticmethod
    def _last_map(x):
        # 仅取最后一级特征图
        if isinstance(x, (list, tuple)):
            return x[-1]
        return x

    @torch.no_grad()
    def _teacher_last(self, img):
        # 只走 backbone，拿多尺度中的最后一级（通常是 layer4/C5）
        t_feats = self.teacher.backbone(img)  # 直接 backbone
        return self._last_map(t_feats)

    def forward_train(self, img, **data_samples):
        # 1) 先得到 teacher 的最后一级特征（无梯度）
        t_last = self._teacher_last(img)

        # 2) 运行 student 的训练前向，产生任务损失；
        #    同时 hook 会把 student.backbone 的输出放入 feat_bucket
        losses = self.student.forward_train(img=img, **data_samples)

        # 3) 取出 student 的最后一级特征
        s_feats_all = self._feat_bucket.pop('s_backbone_feats', None)
        if s_feats_all is None:
            print('student backbone features is None')
            return losses  # 理论上不会发生，仅做兜底
        s_last = self._last_map(s_feats_all)

        # 4) 形状对齐后做 MSE
        #    a) 空间尺寸对齐到学生
        if t_last.shape[-2:] != s_last.shape[-2:]:
            t_last = F.interpolate(t_last, size=s_last.shape[-2:], mode='bilinear', align_corners=False)
        #    b) 通道不一致时，取前 min(Ct, Cs) 做 MSE（稳定、省事）
        if t_last.shape[1] != s_last.shape[1]:
            c = min(t_last.shape[1], s_last.shape[1])
            kd = F.mse_loss(s_last[:, :c], t_last[:, :c])
        else:
            # print("s: "+ str(s_last.shape))
            # print("t: "+str(t_last.shape))
            kd = F.mse_loss(s_last, t_last)

        losses['loss_kd_backbone_last_mse'] = kd * self.kd_weight
        # effective_kd_weight = F.softplus(self.kd_weight_raw)
        # losses['loss_kd_backbone_last_mse'] = kd * effective_kd_weight
        # print(losses)
        return losses

    def forward(self, *args, **kwargs):
        # 推理阶段直接走学生
        return self.student.forward(*args, **kwargs)



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # model = build_model(
    #     cfg.model,
    #     train_cfg=cfg.get('train_cfg'),
    #     test_cfg=cfg.get('test_cfg'))
    # model.init_weights()
    #
    # state_dict = torch.load('ckp/PGD_pretrain.pth')
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # loaded_keys = set(state_dict.keys()) - set(unexpected)
    # print("==== 成功加载的 keys ====")
    # for i, key in enumerate(loaded_keys):
    #     print(i, key)

    model = build_model(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # 从配置读取
    teacher_ckpt = cfg.get('distill', {}).get('teacher_ckpt', None)
    kd_weight = cfg.get('distill', {}).get('kd_weight', 1.0)
    assert teacher_ckpt is not None, '请在 cfg.distill.teacher_ckpt 指定教师权重路径'

    # teacher 结构与 student 一致
    teacher = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    teacher.init_weights()

    # 只加载到 teacher
    t_state = torch.load(teacher_ckpt, map_location='cpu')
    missing, unexpected = teacher.load_state_dict(t_state, strict=False)
    loaded_keys = set(t_state.keys()) - set(unexpected)
    print("==== 成功加载的 keys ====")
    for i, key in enumerate(loaded_keys):
        print(i, key)

    # 用蒸馏包装器替换 student
    model = DistillWrapper(student=model, teacher=teacher, kd_weight=kd_weight)


    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
