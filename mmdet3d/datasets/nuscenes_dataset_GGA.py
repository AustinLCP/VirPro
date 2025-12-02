# Copyright (c) OpenMMLab. All rights reserved.
# Customized nuScenes dataset for GGA (region-grow + ground) fields.
#
# This class extends the official NuScenesDataset to attach and load per-sample
# GGA annotations produced by your converter, aligning **exactly** with the keys
# requested in your current config (Collect3D_GGA keys), namely:
#   'GGA_boxes_img', 'GGA_lidar2img', 'GGA_init_pseudo_labels',
#   'GGA_bdry_masks', 'GGA_in_box_points'
#
# It also supports additional GGA keys often saved by the converter (mask2d,
# mask_depth, mask_boundary, mask_valid, num_points_in_box2d). Missing files or
# fields are handled gracefully to avoid KeyErrors in the pipeline.

from __future__ import annotations
import os.path as osp
import pickle
from typing import Any, Dict, Optional

import numpy as np
from pyquaternion import Quaternion

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.builder import DATASETS


def _Rt(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = T
    return M


@DATASETS.register_module()
class NuScenesDataset_GGA_train(NuScenesDataset):
    """nuScenes dataset with GGA attachments aligned to your config.

    Args:
        gga_dir (str, optional): Directory of per-sample GGA .pkl files.
            Default: ``data_root/nuscenes_GGA_split_file``.
        gga_camera (str, optional): Camera key used for filename mapping
            (e.g., 'CAM_FRONT'). Default: 'CAM_FRONT'.
        **kwargs: Passed to parent ``NuScenesDataset``.
    """

    # Keep class order aligned with your 10-class setup in config
    METAINFO = dict(
        classes=(
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        )
    )

    def __init__(self, gga_dir: Optional[str] = None, gga_camera: str = 'CAM_FRONT', **kwargs):
        self.gga_camera = gga_camera
        super().__init__(**kwargs)
        if gga_dir is None:
            gga_dir = osp.join(self.data_root, 'nuscenes_GGA_split_file')
        self.gga_dir = gga_dir

    # ---------- paths & utils ----------
    def _build_gga_path(self, token: str) -> str:
        # Your converter uses: GGA_nus_{token}_{camera}.pkl
        fname = f'GGA_nus_{token}_{self.gga_camera}.pkl'
        return osp.join(self.gga_dir, fname)

    def _safe_load_pickle(self, path: str) -> Optional[dict]:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    # Compute lidar->img projection for the requested camera from data_infos
    def _compute_lidar2img(self, index: int) -> Optional[np.ndarray]:
        info = self.data_infos[index]
        cams = info.get('cams', None)
        if cams is None or self.gga_camera not in cams:
            return None
        cam_info = cams[self.gga_camera]

        # Extract transforms
        # lidar side
        l2e_R = np.array(info.get('lidar2ego_rotation'), dtype=np.float32)
        l2e_T = np.array(info.get('lidar2ego_translation'), dtype=np.float32)
        if l2e_R.shape == (4,):
            l2e_R = Quaternion(l2e_R).rotation_matrix.astype(np.float32)
        else:
            l2e_R = Quaternion(l2e_R).rotation_matrix.astype(np.float32)

        # ego->global (lidar timestamp)
        e2g_R = Quaternion(np.array(info.get('ego2global_rotation'), dtype=np.float32)).rotation_matrix.astype(np.float32)
        e2g_T = np.array(info.get('ego2global_translation'), dtype=np.float32)

        # camera side
        c2e_R = Quaternion(np.array(cam_info.get('sensor2ego_rotation'), dtype=np.float32)).rotation_matrix.astype(np.float32)
        c2e_T = np.array(cam_info.get('sensor2ego_translation'), dtype=np.float32)
        cam_K = np.array(cam_info.get('cam_intrinsic'), dtype=np.float32)  # 3x3

        lidar2ego = _Rt(l2e_R, l2e_T)
        ego2global = _Rt(e2g_R, e2g_T)
        cam2ego = _Rt(c2e_R, c2e_T)
        ego2cam = np.linalg.inv(cam2ego)

        # In nuScenes, lidar and cams are at same timestamp for a sample token
        # A simplified chain: lidar->ego(lidar)->ego(cam)->cam
        # Strict chain with global cancels out when timestamps match
        lidar2cam = ego2cam @ lidar2ego

        # build 3x4 K [I|0] then compose: P = K @ [R|t] where [R|t] = lidar2cam[:3]
        view = np.eye(4, dtype=np.float32)
        view[:3, :] = lidar2cam[:3, :]
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[:3, :3] = cam_K
        lidar2img = proj @ view
        return lidar2img

    # ---------- API to pipeline ----------
    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = super().get_data_info(index)
        token = info.get('token', None)
        if token is None:
            token = self.data_infos[index].get('token', None)
        if token is not None:
            info['gga_pkl'] = self._build_gga_path(token)
            info['GGA_camera'] = self.gga_camera

        # Ensure `lidar2img` is a single ndarray for the chosen GGA camera,
        # because your loading pipeline indexes it with [...], which fails on list.
        l2img = self._compute_lidar2img(index)
        if l2img is not None:
            # Preserve any pre-existing multi-view matrices to a side key
            if isinstance(info.get('lidar2img', None), (list, tuple)):
                info['lidar2img_all'] = info['lidar2img']
            info['lidar2img'] = l2img  # single 4x4 ndarray expected by loader
        return info

    def get_ann_info(self, index: int) -> Dict[str, Any]:
        ann_info = super().get_ann_info(index)

        # attach GGA if exists
        token = self.data_infos[index].get('token', None)
        if token is None:
            di = self.get_data_info(index)
            token = di.get('token', None)
        if token is None:
            return ann_info

        gga_path = self._build_gga_path(token)
        gga = self._safe_load_pickle(gga_path)
        if gga is None:
            # still attach lidar2img if we can compute it to satisfy config
            l2img = self._compute_lidar2img(index)
            if l2img is not None:
                ann_info['GGA_lidar2img'] = l2img
            return ann_info

        # Merge expected keys (align to config)
        # 1) mandatory keys requested in config
        if 'GGA_boxes_img' in gga:
            # import numpy as np
            # _boxes = np.asarray(gga['GGA_boxes_img'])
            # # 典型异常 1: (1, N, 4) -> (N, 4)
            # if _boxes.ndim == 3 and _boxes.shape[0] == 1:
            #     _boxes = _boxes[0]
            # # 典型异常 2: (N, 4, 1) -> (N, 4)
            # if _boxes.ndim == 3 and _boxes.shape[-1] == 1:
            #     _boxes = _boxes[..., 0]
            # # 典型异常 3: (4, N) -> (N, 4)
            # if _boxes.ndim == 2 and _boxes.shape[0] == 4 and _boxes.shape[1] != 4:
            #     _boxes = _boxes.T
            # # 典型异常 4: list of boxes -> (N, 4)
            # if _boxes.dtype == object:
            #     _boxes = np.vstack(list(_boxes)).astype(np.float32)
            # assert _boxes.ndim == 2 and _boxes.shape[1] == 4, f"bad GGA_boxes_img shape: {_boxes.shape}"
            ann_info['GGA_boxes_img'] = gga['GGA_boxes_img']
        # handle singular/plural variance
        if 'GGA_init_pseudo_labels' in gga:
            ann_info['GGA_init_pseudo_labels'] = gga['GGA_init_pseudo_labels']
        elif 'GGA_init_pseudo_label' in gga:
            ann_info['GGA_init_pseudo_labels'] = gga['GGA_init_pseudo_label']
        if 'GGA_bdry_masks' in gga:
            ann_info['GGA_bdry_masks'] = gga['GGA_bdry_masks']
        if 'GGA_in_box_points' in gga:
            ann_info['GGA_in_box_points'] = gga['GGA_in_box_points']

        # 2) provide lidar2img; prefer file content, otherwise compute
        if 'GGA_lidar2img' in gga:
            ann_info['GGA_lidar2img'] = gga['GGA_lidar2img']
        else:
            l2img = self._compute_lidar2img(index)
            if l2img is not None:
                ann_info['GGA_lidar2img'] = l2img

        # 3) optional extra GGA keys (loader may or may not use them)
        for k in [
            'GGA_mask2d', 'GGA_mask_depth', 'GGA_mask_boundary',
            'GGA_mask_valid', 'GGA_num_points_in_box2d'
        ]:
            if k in gga:
                ann_info[k] = gga[k]

        # Also expose the path/camera for downstream lazy loaders
        ann_info['gga_pkl'] = gga_path
        ann_info['GGA_camera'] = self.gga_camera

        # --- ensure both singular & plural keys exist for loaders ---
        # Some pipelines expect 'GGA_init_pseudo_label' (singular), others use
        # 'GGA_init_pseudo_labels' (plural). We mirror the value to both.
        if 'GGA_init_pseudo_labels' in ann_info and 'GGA_init_pseudo_label' not in ann_info:
            ann_info['GGA_init_pseudo_label'] = ann_info['GGA_init_pseudo_labels']
        if 'GGA_init_pseudo_label' in ann_info and 'GGA_init_pseudo_labels' not in ann_info:
            ann_info['GGA_init_pseudo_labels'] = ann_info['GGA_init_pseudo_label']

        # --- provide default 'difficulty' for nuScenes (not present by default) ---
        if 'difficulty' not in ann_info:
            try:
                n = len(ann_info.get('gt_labels_3d', []))
            except Exception:
                n = 0
            import numpy as _np
            ann_info['difficulty'] = _np.zeros((n,), dtype=_np.int32)
        return ann_info
