import argparse
from os import path as osp

from tools.data_converter import indoor_converter as indoor
# from tools.data_converter import kitti_converter as kitti
from tools.data_converter import kitti_converter_gga as kitti
from tools.data_converter import nuscenes_converter as nuscenes
#from tools.data_converter import nuscenes_converter_gga as nuscenes
# from tools.data_converter.create_gt_database import (
#     GTDatabaseCreater, create_groundtruth_database)
from tools.data_converter.create_gt_database_gga import (
    GTDatabaseCreater, create_groundtruth_database)


## our addition
# from tools.data_converter.create_gt_database import(
#     visual_and_calculate_boxiou)


def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train_GGA.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val_GGA.pkl')
    info_trainval_path = osp.join(root_path, f'{info_prefix}_infos_trainval_GGA.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)

    create_groundtruth_database(
        'KittiDataset_GGA',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train_GGA.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix='nuscenes',
                       version='v1.0-trainval',
                       max_sweeps=10):
    """Data preparation pipeline for nuScenes (GGA version).

    This function:
      1) Creates nuScenes info files (train/val or test) via nuscenes_converter.
      2) Builds a GGA-style ground-truth database from the TRAIN info.

    Args:
        root_path (str): nuScenes 数据根目录 (包含 `samples/`, `sweeps/`, `maps/`, `v1.0-*` 等)。
        info_prefix (str): 生成的 info 与数据库前缀，默认 'nuscenes'。
        version (str): nuScenes 版本，如 'v1.0-trainval' / 'v1.0-mini' / 'v1.0-test'。
        max_sweeps (int): 每个 keyframe 累积的 sweeps 数，默认 10。
        used_classes (list[str] | None): 限定要写入 db 的类别；None 表示全部。
        with_mask (bool): 是否额外导出/使用 2D mask（通常为 False）。

    Notes:
        - 对于 'v1.0-test'，由于无标注，只会生成 test infos，不会创建 GT database。
        - 生成的 GT 数据库目录为：{root_path}/{info_prefix}_gt_database_GGA/
          索引文件为：{root_path}/{info_prefix}_dbinfos_train_GGA.pkl
    """

    nuscenes.create_nuscenes_infos(root_path=root_path, info_prefix=info_prefix, version=version, max_sweeps=max_sweeps)

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val_GGA.pkl')
    info_trainval_path = osp.join(root_path, f'{info_prefix}_infos_trainval_GGA.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    nuscenes.export_2d_annotation(root_path, info_train_path,version=version)
    nuscenes.export_2d_annotation(root_path, info_val_path,version=version)
    nuscenes.export_2d_annotation(root_path, info_trainval_path,version=version)
    nuscenes.export_2d_annotation(root_path, info_test_path,version=version)


    is_test = 'test' in version
    if is_test:
        print('[nuScenes] Test split detected; skip GT database creation (no annotations).')
        return

    create_groundtruth_database(
        dataset_class_name='NuScenesDataset',
        data_path=root_path,
        info_prefix=info_prefix,
        info_path=info_train_path,
        # 下面保持默认：会在 data_path 下生成
        #   {info_prefix}_gt_database_GGA/    (实例点云库)
        #   {info_prefix}_dbinfos_train_GGA.pkl
        relative_path=True,
        with_mask=False  # nuScenes 流程默认不需要 2D 实例 mask
    )  # 针对 NuScenes 的 pipeline 会加载多帧 sweeps 并写出 dbinfos。:contentReference[oaicite:3]{index=3}

    print('[nuScenes] Done. GT database saved (GGA style).')
    

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root_path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0-trainval',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--num-points',
    type=int,
    default=-1,
    help='Number of points to sample for indoor datasets.')
parser.add_argument(
    '--out_dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='nuscenes')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            with_plane=args.with_plane)
    elif args.dataset == 'nuscenes':
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag if args.extra_tag else 'nuscenes',
            version=args.version,  # 例如: 'v1.0-trainval' 或 'v1.0-test'
            max_sweeps=getattr(args, 'max_sweeps', 10)
        )

    # elif args.dataset == 'sunrgbd':
    #     sunrgbd_data_prep(
    #         root_path=args.root_path,
    #         info_prefix=args.extra_tag,
    #         num_points=args.num_points,
    #         out_dir=args.out_dir,
    #         workers=args.workers)
