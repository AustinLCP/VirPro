import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

train_3D_mapping_file_path = Path('./data/kitti/data_file/train_mapping.txt')
kitti_3D_rand_file_path = Path('./data/kitti/data_file/train_rand.txt')
# train_3D_file_path = Path('./data/kitti/data_file/split/train.txt')
val_3D_file_path = Path('./data/kitti/data_file/split/val.txt')

kitti_raw_data_dir = Path('./data/kitti/raw_data')
save_kitti_raw_file_name = Path('./data/kitti/data_file/split/train_raw.txt')


# raw_file_list: len = 42478
#   [['..\\2011_09_26\\2011_09_26_drive_0001_sync\\image_02\\data\\0000000000.png',
#       '..\\2011_09_26\\2011_09_26_drive_0001_sync\\image_03\\data\\0000000000.png']...]

# raw_file_velo_list: len = 42478
#   [['..\\2011_09_26\\2011_09_26_drive_0001_sync\\image_02\\data\\0000000000.png',
#       '..\\2011_09_26\\2011_09_26_drive_0001_sync\\velodyne_points\\data\\0000000000.bin']...]
# 提取数据集中所有的文件
def build_all_files():
    raw_file_list = []
    raw_file_velo_list = []
    data_dir = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
    for d1 in data_dir:
        data_dir2 = sorted(os.listdir(os.path.join(kitti_raw_data_dir, d1)))
        for d2 in data_dir2:
            if os.path.exists(os.path.join(kitti_raw_data_dir, d1, d2, 'velodyne_points')):
                im_ind = sorted(os.listdir(os.path.join(kitti_raw_data_dir, d1, d2, 'velodyne_points', 'data')))

                im2 = [os.path.join(kitti_raw_data_dir, d1, d2, 'image_02', 'data', i.replace('bin', 'png')) for i in im_ind]
                im3 = [os.path.join(kitti_raw_data_dir, d1, d2, 'image_03', 'data', i.replace('bin', 'png')) for i in im_ind]
                im_velo = [os.path.join(kitti_raw_data_dir, d1, d2, 'velodyne_points', 'data', i) for i in im_ind]

                im_name = np.concatenate([np.array(im2).reshape(-1, 1), np.array(im3).reshape(-1, 1)], axis=1)
                im_name_velo = np.concatenate([np.array(im2).reshape(-1, 1), np.array(im_velo).reshape(-1, 1)],
                                               axis=1)

                raw_file_list.extend(list(im_name))
                raw_file_velo_list.extend(list(im_name_velo))
    raw_file_list = np.array(raw_file_list)
    raw_file_velo_list = np.array(raw_file_velo_list)
    print('build raw files, done')

    return raw_file_list, raw_file_velo_list


# val_set: len = 45
#   [''2011_09_28_drive_0100_sync'', ''2011_09_26_drive_0032_sync'', ''2011_09_26_drive_0009_sync'',...]

# val_3D_mapping  len = 3769
    # [['2011_09_26', '2011_09_26_drive_0101_sync', '0000000528'],
    #  ['2011_09_26', '2011_09_26_drive_0086_sync', '0000000108']...]
# 根据val.txt 从数据集中选出所有用于验证的数据
def build_train_val_set():
    train_mapping = np.loadtxt(train_3D_mapping_file_path, dtype=str)  # len = 7481
    kitti_rand = np.loadtxt(kitti_3D_rand_file_path, delimiter=',')  # len = 7481
    # train_3D = np.loadtxt(train_3D_file_path).astype(np.uint16)
    val_3D = np.loadtxt(val_3D_file_path).astype(np.uint16)  # 把val_3D.txt中每一行的数字作为一个元素存入list, len = 3769

    # train_3D_mapping = train_mapping[(kitti_rand[train_3D]-1).astype(np.uint16)]

    # 根据验证集索引从随机映射数组 kitti_rand 中取出相应的元素，并对这些元素进行了偏移和类型转换，
    # 用以适配 train_mapping 中的索引。然后通过这些索引从 train_mapping 中获取相应的验证集映射
    val_3D_mapping = train_mapping[(kitti_rand[val_3D]-1).astype(np.uint16)]
    # train_set = set([i[1] for i in train_3D_mapping])
    val_set = set([i[1] for i in val_3D_mapping])

    print('remove val scenes in raw data, done')

    # return list(train_set), list(val_set), train_3D_mapping, val_3D_mapping
    return list(val_set), val_3D_mapping


# train_files: len = 33530
#   [['..\\2011_09_30\\2011_09_30_drive_0033_sync\\image_02\\data\\0000001028.png',
#       ..\\2011_09_30\\2011_09_30_drive_0033_sync\\velodyne_points\\data\\0000001028.bin']...]
# 在数据集所有的文件中(raw_file_list), 排除掉所有用于验证的数据文件(val_set),从而得到用于训练的数据
def build_train_files(name_velo_file, val_set):
    good_ind = np.ones(len(name_velo_file)).astype(np.bool_)
    scene_name = np.array([i[0].split('\\')[-4] for i in name_velo_file])

    for f in tqdm(sorted(val_set)):
        good_ind[scene_name == f] = False

    train_files = name_velo_file[good_ind]
    np.random.shuffle(train_files)

    print('build train files, done')

    return train_files


# train_add_calib_file: len = 33530
#   [['..\\2011_09_30\\2011_09_30_drive_0033_sync\\image_02\\data\\0000001028.png',
#      '..\\2011_09_30\\2011_09_30_drive_0033_sync\\velodyne_points\\data\\0000001028.bin',
#       '..\\2011_09_30\\calib_cam_to_cam.txt',
#       '..\\2011_09_30\\calib_velo_to_cam.txt']...]
# 加上对应的calib_cam_to_cam.txt 和 calib_velo_to_cam.txt
def add_calib_file(name_velo_file):
    cam_path_1_list = [os.path.join(*(f[0].split('\\')[:-4]), 'calib_cam_to_cam.txt') for f in name_velo_file]
    cam_path_2_list = [os.path.join(*(f[0].split('\\')[:-4]), 'calib_velo_to_cam.txt') for f in name_velo_file]
    add_calib_file = np.concatenate([name_velo_file,
                                      np.array(cam_path_1_list).reshape(-1, 1),
                                      np.array(cam_path_2_list).reshape(-1, 1)], axis=1)

    print('add calib files, done')

    return add_calib_file


if __name__ == '__main__':
    raw_file_list, raw_file_velo_list = build_all_files()
    # train_set, val_set, train_3D_mapping, val_3D_mapping = build_train_val_set()
    val_set, val_3D_mapping = build_train_val_set()
    train_files = build_train_files(raw_file_velo_list, val_set)
    train_add_calib = add_calib_file(train_files)

    np.savetxt(save_kitti_raw_file_name, train_add_calib, fmt='%s')







