# nuScenes config for GGA, adapted from gga_kitti_config.py
# NOTE: Adjust class list, voxel/grid settings, and dataset type names
# to match your project if they differ.
custom_imports = dict(
    imports=['mmdet3d.datasets.nuscenes_dataset_GGA'],
    allow_failed_imports=False,
)



voxel_size = [0.1, 0.1, 0.2]
# Common nuScenes range used by CenterPoint/SECOND-style models
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

model = dict(
    type='GGA',
    pts_voxel_layer=dict(
        max_num_points=5,
        voxel_size=voxel_size,
        max_voxels=(30000, 60000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        # For nuScenes typical z bins = (z_max - z_min) / vz = (3 - (-5)) / 0.2 = 40
        # x,y based on grid_size below
        sparse_shape=[40, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead_GGA',
        in_channels=sum([256, 256]),
        # nuScenes 10-class setup
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=2, class_names=['barrier', 'traffic_cone']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=1, class_names=['pedestrian']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=point_cloud_range,
            max_num=100,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7,
            pc_range=point_cloud_range[:2],
        ),
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', alpha=0.),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_center=dict(type='MarginL1Loss', reduction='mean'),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[0.5, 0.5, 0.5, 0.5, 0.5],
            margin_weights=[1.0, 1.0],
        ),
    ),
    test_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            # class-wise min_radius; tune as needed
            min_radius=[4, 12, 12, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=512,
            nms_thr=0.2,
        ),
    )
)

# ===== Dataset =====
dataset_type = 'NuScenesDataset_GGA_train'  # adapt to your dataset class name
data_root = 'data/nuscenes/'
# Standard nuScenes 10 classes (lowercase to match many implementations)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

input_modality = dict(use_lidar=True, use_camera=True)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4, file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        # with_bbox=True,
        with_gga=True,
        file_client_args=file_client_args,
    ),
    # Optional: nuScenes frequently uses multi-sweep points, add if needed
    # dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=4, pad_empty_sweeps=True, remove_close=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter_GGA', point_cloud_range=point_cloud_range, num_points_range=15),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D_GGA', class_names=class_names),
    dict(
        type='Collect3D_GGA',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
              # GGA inputs
              'GGA_boxes_img', 'GGA_lidar2img', 'GGA_init_pseudo_labels',
              'GGA_bdry_masks', 'GGA_in_box_points']
    ),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4, file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1., 1.], translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points', 'img']),
        ],
    ),
]

# for visualization/eval scripts that call dataset.show()
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4, file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points', 'img']),
]

data = dict(
    samples_per_gpu=16,  # nuScenes is heavier; tune by GPU memory
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file = data_root + 'nus_infos_train.pkl',
            # split='train',  # or 'trainval' depending on how your Dataset class uses the split
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            gga_dir=data_root + 'nuscenes_GGA_split_file',
            gga_camera='CAM_FRONT',
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file = data_root + 'nus_infos_val.pkl',
        # split='val',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file = data_root + 'nus_infos_val.pkl',
        # split='val',  # change to 'test' for test-set inference
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
    ),
)

evaluation = dict(interval=5, pipeline=eval_pipeline)

optimizer = dict(type='AdamW', lr=0.0015, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4,
)

momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

runner = dict(type='EpochBasedRunner', max_epochs=120)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='GGA',
                name='NuScenes_GGA_train',
                config=dict(lr=0.0015, batch_size=8, max_epochs=120, model='GGA'),
            ),
            by_epoch=True,
        ),
    ],
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/nuscenes_GGA'
load_from = None
resume_from = None
workflow = [('train', 1)]

gpu_ids = range(0, 1)

# default runtime
opencv_num_threads = 0
