# tools/eval.py
from __future__ import division
import argparse
import os
import os.path as osp
import torch
from numba import cuda as nb_cuda
# if torch.cuda.is_available():
#     torch.cuda.set_device(0)                 # 显式绑定 torch 到 0 号（映射后的）GPU
#     dev = torch.cuda.current_device()
#     nb_cuda.select_device(dev)               # 让 numba 绑定同一块卡
#     _ = nb_cuda.current_context()            # 强制创建 numba CUDA 上下文
#     print(f"[eval] torch device={dev}, numba bound and context created")
import mmcv
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.apis import single_gpu_test
from mmcv.parallel import DataContainer
from tqdm import tqdm
import wandb
import re

def parse_epoch_from_ckpt(ckpt_path: str) -> int:
    name = os.path.basename(ckpt_path)
    m = re.search(r'epoch[_\-]?(\d+)', name)
    if m:
        return int(m.group(1))
    # 兜底：从文件名里的第一段数字取 epoch
    m2 = re.search(r'(\d+)', name)
    return int(m2.group(1)) if m2 else 0




def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints in a folder (mmdet3d 1.0.0rc7)')
    parser.add_argument('config', help='config file path, e.g. configs/gga/gga_pdg.py')
    parser.add_argument('ckpt_dir', help='directory containing .pth checkpoints')
    parser.add_argument('--out-dir', default='eval_results', help='dir to save eval results')
    # parser.add_argument('--device', default='cuda:0', help='device, e.g. cuda:0 or cpu')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='dataset split to evaluate on')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='dataloader workers per gpu')
    parser.add_argument('--samples-per-gpu', type=int, default=1, help='dataloader batch size per gpu')
    return parser.parse_args()


def load_student_weights(model, ckpt_path, device='cuda:0'):
    """
    仅加载 checkpoint 中以 'student.' 开头的参数，
    去掉前缀后加载进 model（strict=False）
    """
    print(f"→ Load & strip prefix 'student.': {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    new_state = {}
    keep_cnt = 0
    for k, v in state_dict.items():
        if k.startswith('student.'):
            # 去掉 'student.' 前缀
            new_k = k[len('student.'):]
            new_state[new_k] = v
            keep_cnt += 1

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"   kept {keep_cnt} keys; missing={len(missing)}, unexpected={len(unexpected)}")

    if isinstance(checkpoint, dict):
        meta = checkpoint.get('meta', {})
        if meta and 'CLASSES' in meta:
            model.CLASSES = meta['CLASSES']

    model.to(device)
    model.eval()
    return model


def normalize_test_pipeline_to_tensor(cfg, split='val'):
    """展开 MultiScaleFlipAug，让 img 变成 tensor 而不是 list。"""
    pl = cfg.data[split]['pipeline']
    # 典型结构: [LoadImageFromFileMono3D, {type=MultiScaleFlipAug, transforms=[...]}]
    new_pl = []
    for step in pl:
        if isinstance(step, dict) and step.get('type') == 'MultiScaleFlipAug':
            # 只取内部 transforms；通常已经包含 Normalize/Pad/DefaultFormatBundle3D/Collect3D(keys=['img'])
            inner = step.get('transforms', [])
            new_pl.extend(inner)
        else:
            new_pl.append(step)
    cfg.data[split]['pipeline'] = new_pl


def single_gpu_test_compat(model, data_loader, show=False):
    model.eval()
    results = []
    prog_bar = tqdm(total=len(data_loader), desc='Inference', ncols=80)

    device = next(model.parameters()).device

    for data in data_loader:
        # -------- 1) img 解包 --------
        img = data.get('img', None)
        if isinstance(img, DataContainer):
            img = img.data[0]             # 可能是 tensor 或 list[tensor]
        if isinstance(img, list):
            assert len(img) >= 1
            img = img[0]                  # 退化为单尺度/单增强
        if not torch.is_tensor(img):
            raise TypeError(f'Expect Tensor for img, got {type(img)}')

        # [1,C,H,W]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(device, non_blocking=True)

        # -------- 2) img_metas 解包 --------
        img_metas = data.get('img_metas', None)
        if isinstance(img_metas, DataContainer):
            img_metas = img_metas.data[0]   # 可能是 list[dict] 或 dict
        if isinstance(img_metas, list):
            assert len(img_metas) >= 1
            img_metas = img_metas[0]        # 取第一份增强的 meta
        if not isinstance(img_metas, dict):
            raise TypeError(f'Expect dict for img_metas, got {type(img_metas)}')

        # 保险：补 batch_input_shape（有些模型里会用到）
        if 'batch_input_shape' not in img_metas:
            h, w = img.shape[-2], img.shape[-1]
            img_metas['batch_input_shape'] = (h, w)

        # —— 规范/兜底 img_metas['scale_factor'] 为「标量」——
        sf = img_metas.get('scale_factor', None)

        def _to_np1d(x):
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, (list, tuple)):
                return np.array(x, dtype=np.float32)
            try:
                return np.array(x, dtype=np.float32)
            except Exception:
                return None

        sf_np = _to_np1d(sf)

        if sf_np is None:
            # 没有就根据 ori_shape / img_shape 反推
            if 'ori_shape' in img_metas and 'img_shape' in img_metas:
                oh, ow = img_metas['ori_shape'][:2]
                ih, iw = img_metas['img_shape'][:2]
                sw = iw / float(ow) if ow else 1.0
                sh = ih / float(oh) if oh else 1.0
            else:
                sw = sh = 1.0
        else:
            # 有值则解析出 sw, sh
            if sf_np.size == 4:
                sw, sh = float(sf_np[0]), float(sf_np[1])
            elif sf_np.size == 2:
                sw, sh = float(sf_np[0]), float(sf_np[1])
            elif sf_np.size == 1:
                sw = sh = float(sf_np.reshape(-1)[0])
            else:
                sw = sh = 1.0

        # 统一成「标量」
        if np.isfinite(sw) and np.isfinite(sh):
            if abs(sw - sh) < 1e-6:
                scalar = float(sw)
            else:
                # 非等比缩放时用平均值兜底（避免维度冲突先跑通）
                scalar = float((sw + sh) / 2.0)
        else:
            scalar = 1.0

        img_metas['scale_factor'] = scalar

        # -------- 3) 直接走 simple_test --------
        with torch.no_grad():
            # simple_test(img: Tensor[B,C,H,W], img_metas: List[Dict], rescale: bool)
            out = model.simple_test(img, [img_metas], rescale=True)

        # out 通常是 List[one_image_result]
        if isinstance(out, list):
            results.extend(out)
        else:
            results.append(out)

        prog_bar.update(1)

    return results


def evaluate_one(cfg, ckpt_path, args):
    print(f"\n========== Evaluating: {osp.basename(ckpt_path)} ==========")

    normalize_test_pipeline_to_tensor(cfg, args.split)

    # 1) 构建数据集 & dataloader（沿用 config 里的 pipeline）
    dataset = build_dataset(cfg.data[args.split])
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,  # 强制 1
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # 2) 构建模型（测试配置）
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 3) 只加载 student→backbone 的权重
    model = load_student_weights(model, ckpt_path)
    # 4) 推理
    outputs = single_gpu_test_compat(model, data_loader, show=False)

    # 5) 评估（rc7: 直接 dataset.evaluate）
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # evaluation 配置里常有 interval 等非评估字段，剔除一下
    for bad in ['interval', 'save_best', 'rule', 'metric_options']:
        eval_kwargs.pop(bad, None)

    results = dataset.evaluate(outputs, **eval_kwargs)
    print(f"Results for {osp.basename(ckpt_path)}:\n{results}")

    # 6) 保存结果
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = osp.join(args.out_dir, osp.basename(ckpt_path).replace('.pth', '_eval.txt'))
    with open(out_file, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    if wandb.run is not None:
        epoch = parse_epoch_from_ckpt(ckpt_path)
        metrics = {f"{args.split}/{k}": v for k, v in results.items()}  # 例如 "val/img_bbox/..."
        # 同时记录一个 "epoch" 字段，并将 step 显式设为 epoch
        wandb.log({"epoch": epoch, **metrics}, step=epoch, commit=True)

    return results


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # === 初始化 wandb ===
    if 'log_config' in cfg and any(h.get('type') == 'WandbLoggerHook' for h in cfg.log_config['hooks']):
        wandb_hook = next(h for h in cfg.log_config['hooks'] if h.get('type') == 'WandbLoggerHook')
        init_kwargs = wandb_hook.get('init_kwargs', {})
        wandb.init(**init_kwargs)
    else:
        wandb.init(project='GGA', name='Eval_Run')
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    # 收集 ckpt
    ckpts = [
        osp.join(args.ckpt_dir, f)
        for f in os.listdir(args.ckpt_dir)
        if f.endswith('.pth')
    ]
    ckpts.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    if not ckpts:
        print(f"No .pth found in {args.ckpt_dir}")
        return

    all_results = {}
    for p in ckpts:
        res = evaluate_one(cfg, p, args)
        all_results[osp.basename(p)] = res

    # 汇总
    summary = osp.join(args.out_dir, 'summary.txt')
    with open(summary, 'w') as f:
        for name, res in all_results.items():
            f.write(f"\n===== {name} =====\n")
            for k, v in res.items():
                f.write(f"{k}: {v}\n")

    print(f"\nAll done. Results saved to: {args.out_dir}")

    if wandb.run is not None:
        wandb.finish()
        print("Results recorded in wandb")


if __name__ == '__main__':
    main()
