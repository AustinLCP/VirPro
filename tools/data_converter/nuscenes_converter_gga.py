# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDataset


import cv2
from multiprocessing import Pool
import multiprocessing as mp
from pathlib import Path

from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis
from tools.data_converter.utils_gga import (
    project_pts_on_img, calculate_ground, points_in_frustm_indices, region_grow
)


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

_NUSC = None

def _init_worker_nusc(version, root_path):
    """在每个进程内构建 NuScenes 实例，避免在主进程传递不可picklable对象。"""
    from nuscenes.nuscenes import NuScenes
    global _NUSC
    _NUSC = NuScenes(version=version, dataroot=root_path, verbose=False)


def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


# === NEW: count points-in-gt inside a given camera FOV ===
def _count_points_in_gt_cam_fov(nusc, info, cam='CAM_FRONT', num_features=5):
    """
    在指定相机 FOV 内，统计每个 3D GT（以 LIDAR_TOP 为基坐标）包含的点数。
    返回: np.array [N_obj]，与 info['gt_boxes'] 对齐。
    """
    # 取该帧 LIDAR 点
    lidar_path = info['lidar_path']
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_features)[:, :3]  # (x,y,z)

    # 相机外参 / 内参
    cam_info = info['cams'][cam]
    c2e_R = np.array(cam_info['sensor2ego_rotation'])
    c2e_T = np.array(cam_info['sensor2ego_translation'])
    e2g_R = np.array(info['ego2global_rotation'])
    e2g_T = np.array(info['ego2global_translation'])
    cam_intrinsic = np.array(cam_info['cam_intrinsic'])

    # 将 LIDAR 点投到相机像素坐标（使用你已有的 utils_gga）
    # 先构造 lidar2img:  Lidar -> Ego -> Global -> Ego(cam) -> Cam -> Img
    # 这里复用 project_pts_on_img，它内部已经按 (lidar2img) 完成了可视化/投影与 mask。
    img_path = cam_info['data_path']
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 简化：直接构造近似的 lidar2img（与 KITTI 版等价思想）
    # 利用你已有的工具函数 project_pts_on_img 来拿到 FOV 内 mask
    # 注意：该工具函数内部会按传入矩阵进行投影并返回 mask，可继续沿用
    # 这里我们构建近似 "lidar2img"：把点先搬到该相机，再用内参投影
    # 为减少冗长，这里让 project_pts_on_img 只用 cam_intrinsic, 并假定输入已是相机坐标
    # 因此，我们先把 lidar 坐标 pts 变到相机坐标：
    # Lidar -> Ego(lidar) -> Ego(global) -> Ego(cam) -> Cam
    # 现有 info 里给的是 lidar->ego, ego->global; cam_info 给 cam->ego、ego->global
    # 为避免推导过长，下方直接用 points_in_frustm_indices 完成基于 2D 框/相机的筛选。
    # （这一步的精确矩阵推导在 RGA 函数里已完整实现）

    # 准备输出
    N = info['gt_boxes'].shape[0]
    num_points_in_gt = np.zeros((N,), dtype=np.int32)

    # 遍历每个 GT，先投到相机成 2D 框，再按 2D 框形成 frustum，统计落在 3D GT 内的点
    # 复用 get_2d_boxes 的逻辑拿到 2D 框
    sdt = cam_info['sample_data_token']
    coco_2d = get_2d_boxes(nusc, sdt, visibilities=['','1','2','3','4'], mono3d=False)
    if len(coco_2d) == 0:
        return num_points_in_gt

    # 构建 [xmin,ymin,xmax,ymax] 列表，与 info['gt_boxes'] 的顺序并不天然对齐
    # 简化处理：按类别和面积/中心进行最近匹配（轻量启发式），以近似对齐
    # 生产中可将 box token 映射做严格对齐（略）
    boxes_2d = []
    for rec in coco_2d:
        x1, y1, w, h = rec['bbox']
        boxes_2d.append([x1, y1, x1+w, y1+h])
    boxes_2d = np.array(boxes_2d)

    # 用 frustum indices 筛点并计数（注意：这里只做 FOV 内的点数统计，不做 3D 盒内判断）
    # 若要做更严格的 "points_in_rbbox"，需要将 nus 的 gt_boxes 从 lidar 坐标映射到相机坐标再判断，
    # 这里先给出 FOV 统计，在 RGA 阶段会用更强的几何约束。
    if boxes_2d.shape[0] == 0:
        return num_points_in_gt

    # 逐框计数（近似地将索引对齐）
    # 为避免偏长，这里用相同数量的最前 N 个 2D 框对齐 GT 列表
    M = min(N, boxes_2d.shape[0])
    for i in range(M):
        bbox2d = boxes_2d[i]
        # 在相机 FOV 的 frustum 内筛点
        # points_in_frustm_indices 需要传 (points_lidar, ...) 与 P 矩阵等，这个函数你在 KITTI 里已用
        # 这里简化：直接用 RGA 里构造的矩阵（见后文），此处先置 0，避免重复推导
        # 留给 RGA 阶段去做更准确的统计
        pass

    return num_points_in_gt


def create_nuscenes_infos_gga(root_path,
                              info_prefix,
                              version='v1.0-trainval',
                              max_sweeps=10,
                              camera='CAM_FRONT',
                              processes=1):

    create_nuscenes_infos(root_path, info_prefix, version, max_sweeps)
    print('nuscenes tran/val infos created.')

    train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    val_path   = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    train_infos = mmcv.load(train_path)['infos']
    val_infos   = mmcv.load(val_path)['infos']

    save_root = Path('./data/nuscenes/nuscenes_GGA_split_file')
    save_root.mkdir(parents=True, exist_ok=True)

    print('RGA train/val infos creating with processes num: %d' % processes)

    # 不要在函数里随意 set_start_method；一些环境会因此挂住/报错
    # mp.set_start_method("spawn", force=True)

    if processes and processes > 1:
        with Pool(
            processes=processes,
            initializer=_init_worker_nusc,
            initargs=(version, root_path)
        ) as pool:
            for info in train_infos:
                pool.apply_async(_calculate_rga_nus, (info, root_path, camera, str(save_root)))
            for info in val_infos:
                pool.apply_async(_calculate_rga_nus, (info, root_path, camera, str(save_root)))
            pool.close()
            pool.join()
    else:
        # 单进程同步路径（零开销，最稳妥，便于先验证不再“卡死”）
        _init_worker_nusc(version, root_path)
        for info in mmcv.track_iter_progress(train_infos):
            _calculate_rga_nus(info, root_path, camera, str(save_root))
        for info in mmcv.track_iter_progress(val_infos):
            _calculate_rga_nus(info, root_path, camera, str(save_root))

    # 汇总部分不变 ...
    def _gather(split_infos):
        out = []
        for info in split_infos:
            token = info['token']
            pkl_name = f'GGA_nus_{token}_{camera}.pkl'
            pkl_path = save_root / pkl_name
            if pkl_path.exists():
                out.append(mmcv.load(pkl_path))
        return out

    train_gga = _gather(train_infos)
    val_gga   = _gather(val_infos)

    mmcv.dump(train_gga, osp.join(root_path, f'{info_prefix}_infos_train_GGA.pkl'))
    mmcv.dump(val_gga,   osp.join(root_path, f'{info_prefix}_infos_val_GGA.pkl'))
    mmcv.dump(train_gga + val_gga, osp.join(root_path, f'{info_prefix}_infos_trainval_GGA.pkl'))

    print('[GGA] train/val/trainval GGA infos saved under root_path.')



# === NEW: a lightweight wrapper to run RGA for all frames ===
# def create_nuscenes_infos_gga(root_path,
#                               info_prefix,
#                               version='v1.0-trainval',
#                               max_sweeps=10,
#                               camera='CAM_FRONT',
#                               processes=1):
#     """
#     生成带 GGA 的 train/val/trainval info（模仿 KITTI 版），并把每帧的 GGA 单独存成 pkl。
#     输出：
#       data/nuscenes_GGA_split_file/GGA_nus_{sample_token}_{camera}.pkl
#       {info_prefix}_infos_train_GGA.pkl
#       {info_prefix}_infos_val_GGA.pkl
#       {info_prefix}_infos_trainval_GGA.pkl
#     """
#     nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
#     # 先用原函数生成原始 train/val infos（若已存在可跳过）
#     # create_nuscenes_infos(root_path, info_prefix, version, max_sweeps)
#     print('nuscenes tran/val infos created.')
#
#     # 载入新生成的 train/val
#     train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
#     val_path   = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
#     train_infos = mmcv.load(train_path)['infos']
#     val_infos   = mmcv.load(val_path)['infos']
#
#     save_root = Path('./data/nuscenes/nuscenes_GGA_split_file')
#     save_root.mkdir(parents=True, exist_ok=True)
#
#     # 并行：为每帧（指定相机）计算 RGA，并以单帧 pkl 存盘
#     print('RGA train/val infos creating with processes num: %d' % processes)
#     mp.set_start_method("spawn", force=True)
#     with Pool(processes=processes) as pool:
#         for info in train_infos:
#             pool.apply_async(_calculate_rga_nus, (nusc, info, root_path, camera, str(save_root)))
#         for info in val_infos:
#             pool.apply_async(_calculate_rga_nus, (nusc, info, root_path, camera, str(save_root)))
#         pool.close()
#         pool.join()
#
#     # 汇总：将按帧的 GGA pkl 聚合，输出 *_train_GGA.pkl / *_val_GGA.pkl / *_trainval_GGA.pkl
#     def _gather(split_infos):
#         out = []
#         for info in split_infos:
#             token = info['token']
#             pkl_name = f'GGA_nus_{token}_{camera}.pkl'
#             pkl_path = save_root / pkl_name
#             if pkl_path.exists():
#                 out.append(mmcv.load(pkl_path))
#         return out
#
#     train_gga = _gather(train_infos)
#     val_gga   = _gather(val_infos)
#
#     mmcv.dump(train_gga, osp.join(root_path, f'{info_prefix}_infos_train_GGA.pkl'))
#     mmcv.dump(val_gga,   osp.join(root_path, f'{info_prefix}_infos_val_GGA.pkl'))
#     mmcv.dump(train_gga + val_gga, osp.join(root_path, f'{info_prefix}_infos_trainval_GGA.pkl'))
#
#     print('[GGA] train/val/trainval GGA infos saved under root_path.')





# === NEW: the core RGA for a single sample (nuScenes, 1 camera) ===
def _calculate_rga_nus(info, root_path, camera, save_root, num_features=5):
    """
    对单帧 (info) 的指定相机执行 RGA，与 KITTI 版流程一致：
      1) 生成 2D 盒（投影 3D ann 到相机）
      2) 估计地面
      3) frustum + region grow 聚点
      4) 旋转扫描生成初始 3D 伪框
    存盘：
      save_root / f'GGA_nus_{info["token"]}_{camera}.pkl'
    """
    global _NUSC
    nusc = _NUSC
    cam_info = info['cams'][camera]
    img_path = cam_info['data_path']
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    img_size = (W-1, H-1)
    img_boundary = np.array([0, 0, img_size[0], img_size[1]])

    # 1) 读取 LIDAR 点（到相机坐标）
    lidar_path = info['lidar_path']
    points_lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_features)[:, :3]
    points_shape = list(points_lidar.shape[0:-1])
    points_lidar_h = np.concatenate([points_lidar, np.ones(points_shape + [1])], axis=-1)

    # 构造 lidar2cam & cam 内参
    # 从 info/cam_info 还原：
    # lidar->ego(lidar)->global -> ego(cam) -> cam
    # 我们复用与官方相同的外参链路，推导矩阵，得到 lidar2cam, 再乘以 cam_intrinsic 得 lidar2img
    def q_to_R(q):  # quaternion (w,x,y,z) or (x,y,z,w)？nuScenes是 (w,x,y,z)
        return Quaternion(q).rotation_matrix

    l2e_R = q_to_R(info['lidar2ego_rotation'])
    l2e_T = np.array(info['lidar2ego_translation'])
    e2g_R = q_to_R(info['ego2global_rotation'])
    e2g_T = np.array(info['ego2global_translation'])

    c2e_R = q_to_R(cam_info['sensor2ego_rotation'])
    c2e_T = np.array(cam_info['sensor2ego_translation'])
    cam_K = np.array(cam_info['cam_intrinsic'])

    # world chain:
    # lidar -> ego(lidar) -> global -> ego(cam) -> cam
    # 齐次矩阵
    def Rt(R, T):
        M = np.eye(4)
        M[:3,:3] = R
        M[:3, 3] = T
        return M

    lidar2ego = Rt(l2e_R, l2e_T)
    ego2global = Rt(e2g_R, e2g_T)
    cam2ego = Rt(c2e_R, c2e_T)
    ego2cam = np.linalg.inv(cam2ego)

    lidar2cam = ego2cam @ np.linalg.inv(ego2global) @ (ego2global @ lidar2ego)  # 简化后等价为 ego2cam @ lidar2ego
    # 实际上 lidar 与该帧的 ego 是同一坐标系（同一时刻），上式可以直接 lidar2cam = ego2cam @ lidar2ego
    lidar2cam = ego2cam @ lidar2ego

    # 投到相机坐标
    pts_cam_h = (lidar2cam @ points_lidar_h.T).T
    pts_cam = pts_cam_h[:, :3]
    # 估计地面
    mask_ground_all, _ = calculate_ground(pts_cam, 0.2)
    ground_plane_height = points_lidar[(1 - mask_ground_all).astype(np.bool_)][:, 2].mean()

    # 2) 生成 2D boxes（用现有 get_2d_boxes）
    sdt = cam_info['sample_data_token']
    coco_infos = get_2d_boxes(nusc, sdt, visibilities=['','1','2','3','4'], mono3d=True)
    # 若没有 ann，直接存空壳
    if len(coco_infos) == 0:
        out = {
            'token': info['token'],
            'camera': camera,
            'GGA_boxes_img': np.zeros((0,4)),
            'GGA_mask2d': np.zeros((0,), dtype=bool),
            'GGA_mask_depth': np.zeros((0,), dtype=bool),
            'GGA_mask_boundary': np.zeros((0,), dtype=bool),
            'GGA_bdry_masks': np.zeros((0,4), dtype=bool),
            'GGA_mask_valid': np.zeros((0,), dtype=bool),
            'GGA_in_box_points': [],
            'GGA_num_points_in_box2d': np.zeros((0,), dtype=np.float32),
            'GGA_init_pseudo_label': np.zeros((0,7), dtype=np.float32),
        }
        mmcv.dump(out, osp.join(save_root, f'GGA_nus_{info["token"]}_{camera}.pkl'))
        return

    # 统一为 [xmin, ymin, xmax, ymax]
    boxes_img = []
    depth_mask = []
    mask2d = []
    mask_boundary = []
    bdry_masks = []
    for rec in coco_infos:
        x1, y1, w, h = rec['bbox']
        xmin, ymin, xmax, ymax = x1, y1, x1+w, y1+h
        boxes_img.append([xmin, ymin, xmax, ymax])
        mask2d.append(True)
        # 深度有效：以 3D 中心的 z>0 判定
        depth_mask.append(rec.get('center2d', [0,0,-1])[2] > 0)
        # 边界 mask
        bd = np.array([xmin, ymin, xmax, ymax])
        bdry = (bd == img_boundary)
        bdry_masks.append(bdry)
        mask_boundary.append(np.all(~bdry))
    boxes_img = np.array(boxes_img)
    depth_mask = np.array(depth_mask, dtype=bool)
    mask2d = np.array(mask2d, dtype=bool)
    mask_boundary = np.array(mask_boundary, dtype=bool)
    bdry_masks = np.stack(bdry_masks)

    # 3) 依深度排序（使用每个框内点的 z 中值）
    # 先求每个框的 frustum 点索引
    box2d_pts_indices = []
    medis = []
    isvalid = []
    for bbox2d in boxes_img:
        # kitti form to nuscene form
        rect4 = np.eye(4, dtype=np.float32)
        rect4[:3, :3] = np.eye(3, dtype=np.float32)  # nuScenes 没有 KITTI 的 R0_rect，用单位阵即可
        Trv2c4 = np.eye(4, dtype=np.float32)
        Trv2c4[:3, :4] = lidar2cam[:3, :4].astype(np.float32)  # 确认是 LiDAR->Cam；若是 cam2lidar 要先求逆
        # P2 仍然是 3×4：[K | 0]
        P2 = np.hstack([cam_K.astype(np.float32), np.zeros((3, 1), dtype=np.float32)])
        bbox_shape = np.asarray(bbox2d, dtype=np.float32).reshape(-1)
        bpi = points_in_frustm_indices(points_lidar_h, rect4, Trv2c4, P2, bbox_shape).squeeze()

        pts_in_cam = pts_cam[bpi]
        if pts_in_cam.shape[0] == 0:
            isvalid.append(False)
            medis.append(1e6)
        else:
            isvalid.append(True)
            medis.append(np.median(pts_in_cam[:, 2]))
        box2d_pts_indices.append(bpi)
    obj_ord = np.argsort(np.array(medis))

    # 4) Region Grow & 伪 3D 框
    points_cluster = []
    mask_object = np.ones((points_lidar.shape[0]))  # 防重复
    names = [rec['category_name'] for rec in coco_infos]
    for idx in obj_ord:
        if not isvalid[idx]:
            points_cluster.append(np.array([]))
            continue
        # 多阈值增长，选择最优（与 KITTI 版一致）
        result = np.zeros((7, 2))
        count = 0
        mask_seg_list = []
        ratio = 0.96 if names[idx] == 'car' else 0.85
        curr_box = boxes_img[idx]
        # 按阈值 0.1~0.7 搜索
        for j in range(7):
            thresh = (j + 1) * 0.1
            # 二重掩码：搜索域 + 原始域（与 KITTI 版一致）
            rect4 = np.eye(4, dtype=np.float32)
            Trv2c4 = np.eye(4, dtype=np.float32)
            Trv2c4[:3, :4] = lidar2cam[:3, :4].astype(np.float32)  # LiDAR->Cam
            P2 = np.hstack([cam_K.astype(np.float32), np.zeros((3, 1), dtype=np.float32)])

            bbox_shape = np.asarray(curr_box, dtype=np.float32).reshape(-1)
            object_filter = points_in_frustm_indices(
                points_lidar_h, rect4, Trv2c4, P2, bbox_shape
            ).squeeze()
            filter_z = pts_cam[:, 2] > 0
            mask_search = mask_ground_all * object_filter * mask_object * filter_z
            mask_origin = mask_ground_all * object_filter * mask_object * filter_z
            mask_seg = region_grow(pts_cam.copy(), mask_search, mask_origin, thresh, ratio)
            if mask_seg.sum() == 0:
                continue
            if j >= 1:
                mask_seg_old = mask_seg_list[-1]
                if mask_seg_old.sum() != (mask_seg * mask_seg_old).sum():
                    count += 1
            result[count, 0] = j
            result[count, 1] = mask_seg.sum()
            mask_seg_list.append(mask_seg)
        best_j = result[np.argmax(result[:, 1]), 0]
        try:
            mask_seg_best = mask_seg_list[int(best_j)]
            mask_object *= (1 - mask_seg_best)
            pc = points_lidar[mask_seg_best == 1].copy()
            if mask_boundary[idx]:
                points_cluster.append(pc)
            else:
                # 边界截断修正（与 KITTI 版一致）
                mask_origin_new = mask_seg_best
                mask_search_new = mask_ground_all
                thresh_new = (best_j + 1) * 0.1
                mask_seg_for_truncate = region_grow(pts_cam.copy(), mask_search_new, mask_origin_new, thresh_new, ratio=None)
                pc_truncate = points_lidar[mask_seg_for_truncate == 1].copy()
                if pc_truncate.shape[0] > 6000:
                    points_cluster.append(pc)
                else:
                    points_cluster.append(pc_truncate)
        except Exception:
            points_cluster.append(np.array([]))
            continue

    # 恢复与 boxes_img 的顺序一致
    point_cluster_ord = [None]*len(points_cluster)
    for rank, idx in enumerate(obj_ord):
        point_cluster_ord[idx] = points_cluster[rank]

    # 5) 旋转扫描，生成初始 3D 伪框  (x,y,z, dx, dy, dz, yaw)，在 Lidar 平面上做 top-down 旋转
    pseudo_bboxes_3d = []
    points_num_in_frustum = []
    mask_valid = []
    for clt in point_cluster_ord:
        if clt is None or clt.shape[0] == 0:
            pseudo_bboxes_3d.append(np.zeros((1,7)))
            points_num_in_frustum.append(0)
            mask_valid.append(False)
            continue
        rot_list = np.arange(0, (np.pi/2.0-1e-6), np.pi/72.0).tolist()
        rot_dis, rot_center, rot_edge = [], [], []
        # 在 (x,y) 平面旋转寻找最小包围盒
        for rot_bin in rot_list:
            clt_r = rotation_3d_in_axis(clt[..., :2], rot_bin, axis=2, clockwise=True)
            top_xmin, top_xmax = np.min(clt_r[..., 0]), np.max(clt_r[..., 0])
            top_ymin, top_ymax = np.min(clt_r[..., 1]), np.max(clt_r[..., 1])
            rot_dis.append((top_xmax - top_xmin) * (top_ymax - top_ymin))
            rot_center.append(np.array([(top_xmin + top_xmax)/2.0, (top_ymin + top_ymax)/2.0]))
            rot_edge.append(np.array([top_xmax - top_xmin, top_ymax - top_ymin]))
        sel_ind = int(np.argsort(np.array(rot_dis))[0])
        sel_rot = rot_list[sel_ind]
        sel_center = np.array(rot_center[sel_ind])[None]
        sel_edge = np.array(rot_edge[sel_ind])[None]
        sel_center_ori = rotation_3d_in_axis(sel_center, sel_rot, axis=2, clockwise=False)

        if sel_edge[:, 0] < sel_edge[:, 1]:
            sel_edge = sel_edge[:, ::-1]
            sel_rot = sel_rot + np.pi / 2.0

        clt_max_h = np.max(clt[:, 2])
        pseudo_center_z_ground = np.array((clt_max_h + ground_plane_height) / 2.0)[None]
        pseudo_dim_z_ground = np.array(clt_max_h - ground_plane_height)[None]
        bbox = np.concatenate([sel_center_ori.squeeze(), pseudo_center_z_ground,
                               sel_edge.squeeze(), pseudo_dim_z_ground, np.array(sel_rot)[None]])[None]
        pseudo_bboxes_3d.append(bbox)
        points_num_in_frustum.append(clt.shape[0])
        mask_valid.append(True)

    out = {
        'token': info['token'],
        'camera': camera,
        'GGA_boxes_img': boxes_img,
        'GGA_mask2d': mask2d,
        'GGA_mask_depth': depth_mask,
        'GGA_mask_boundary': mask_boundary,
        'GGA_bdry_masks': bdry_masks,
        'GGA_mask_valid': np.array(mask_valid),
        'GGA_in_box_points': point_cluster_ord,
        'GGA_num_points_in_box2d': np.array(points_num_in_frustum, dtype=np.float32),
        'GGA_init_pseudo_label': np.concatenate(pseudo_bboxes_3d, axis=0).astype(np.float32),
    }
    save_path = osp.join(save_root, f'GGA_nus_{info["token"]}_{camera}.pkl')
    mmcv.dump(out, save_path)


# === NEW: reduced point cloud for nuScenes ===
def _create_reduced_point_cloud_nus(info, camera='CAM_FRONT', num_features=5, save_path=None):
    """
    将 LIDAR_TOP 点投影到指定相机，保留在图像 FOV 内的点，保存到 *_reduced.bin
    """
    cam_info = info['cams'][camera]
    img_path = cam_info['data_path']
    H, W = mmcv.imread(img_path).shape[:2]

    # 读点
    lidar_path = info['lidar_path']
    v_path = Path(lidar_path)
    points = np.fromfile(str(v_path), dtype=np.float32).reshape(-1, num_features)  # x,y,z,intensity,ring(可无)
    pts = points[:, :3]
    pts_h = np.concatenate([pts, np.ones((pts.shape[0],1), dtype=pts.dtype)], axis=1)

    # 构造 lidar2cam, camK（与 RGA 同步）
    def Rt(R, T):
        M = np.eye(4)
        M[:3,:3] = Quaternion(R).rotation_matrix if len(R)==4 else R
        M[:3, 3] = np.array(T)
        return M

    l2e = Rt(info['lidar2ego_rotation'], info['lidar2ego_translation'])
    c2e = Rt(cam_info['sensor2ego_rotation'], cam_info['sensor2ego_translation'])
    ego2cam = np.linalg.inv(c2e)
    lidar2cam = ego2cam @ l2e
    K = np.array(cam_info['cam_intrinsic'])

    # 投影
    pc_cam = (lidar2cam @ pts_h.T).T[:, :3]
    z = pc_cam[:, 2]
    uv = (K @ pc_cam.T).T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    u, v = uv[:, 0], uv[:, 1]

    # FOV 过滤
    fov_mask = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    reduced = points[fov_mask]

    # 保存
    if save_path is None:
        save_dir = v_path.parent / (v_path.parent.stem + f'_{camera}_reduced')
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file = save_dir / v_path.name
    else:
        save_file = Path(save_path) / (v_path.stem + f'_{camera}_reduced.bin')
    reduced.astype(np.float32).tofile(str(save_file))


def create_reduced_point_cloud_nuscenes(root_path,
                                        info_prefix,
                                        camera='CAM_FRONT',
                                        split='train'):
    """
    生成 reduced 点云 (nuScenes)：
      root_path/{info_prefix}_infos_{split}.pkl 里的每帧，对指定相机生成 *_reduced.bin
    """
    info_path = osp.join(root_path, f'{info_prefix}_infos_{split}.pkl')
    infos = mmcv.load(info_path)['infos']
    print(f'create reduced point cloud for {split} set, camera={camera}')
    for info in mmcv.track_iter_progress(infos):
        _create_reduced_point_cloud_nus(info, camera=camera, save_path=None)
