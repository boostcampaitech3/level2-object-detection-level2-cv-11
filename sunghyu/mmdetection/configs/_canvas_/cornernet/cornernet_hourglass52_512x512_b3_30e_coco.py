_base_ = [
    '_canvas_base_/datasets/coco_detection.py',
    '_canvas_base_/schedules/sgd_cosann.py', '_canvas_base_/default_runtime.py'
]

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# model settings
model = dict(
    type='CornerNet',
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2, # 2 for Hourglass-104
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CornerHead',
        num_classes=10,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=1,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_embedding=dict(
            type='AssociativeEmbeddingLoss',
            pull_weight=0.10,
            push_weight=0.10),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        corner_topk=100,
        local_maximum_kernel=3,
        distance_threshold=0.5,
        score_thr=0.05,
        max_per_img=100,
        nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian')))

# data settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='HueSaturationValue',
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.1),
    dict(
        type='RandomRotate90',
        p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    # dict(
    #     type='RandomCenterCropPad',
    #     crop_size=(511, 511),
    #     ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
    #     test_mode=False,
    #     test_pad_mode=None,
    #     **img_norm_cfg),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False), # img_scale=(511, 511)
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
    ), # 추가
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=True,
        transforms=[
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(
                type='RandomCenterCropPad',
                crop_size=None,
                ratios=None,
                border=None,
                test_mode=True,
                test_pad_mode=['logical_or', 127],
                **img_norm_cfg),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'img_norm_cfg', 'border')),
        ])
]
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'cv_train_3.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'cv_val_3.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
# optimizer = dict(type='Adam', lr=0.001) # lr=0.0005
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[180])
runner = dict(type='EpochBasedRunner', max_epochs=30)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            interval=1000,
            init_kwargs={
                "entity": "canvas11",
                "project": "one-stage-model",
                "name": "CHON_cornernet_hourglass52_512x512_b3_30e"
            }
        )
    ]
)

seed = 2022
gpu_ids = [0]
work_dir = '/opt/ml/detection/baseline/mmdetection/work_dirs/cornernet_hourglass104_mstest_32x3_210e_coco_trash'
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
