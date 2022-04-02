# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (800, 800) #####

albu_train_transforms = [
    dict(
    type='OneOf',
    transforms=[
        dict(type='Flip',p=1.0),
        dict(type='RandomRotate90',p=1.0)
    ],
    p=0.5),
    dict(type='RandomResizedCrop',height=img_scale[0], width=img_scale[1], scale=(0.5, 1.0), p=0.5),
    dict(type='RandomBrightnessContrast',brightness_limit=0.1, contrast_limit=0.15, p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(
    type='OneOf',
    transforms=[
        dict(type='Blur', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='MedianBlur', blur_limit=5, p=1.0),
        dict(type='MotionBlur', p=1.0)
    ],
    p=0.1)
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

##### MOSAIC #####
# https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/tutorials/how_to.md#%E4%BD%BF%E7%94%A8%E9%A9%AC%E8%B5%9B%E5%85%8B%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA
# https://mmdetection.readthedocs.io/en/latest/api.html

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=128.0),
    # dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=128.0),
    dict( 
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
        
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="Pad", pad_to_square=True, pad_val=128.0),

    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train = dict(
    # _delete_ = True,
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'stratified_kfold/basic_v2/cv_train_3.json',
        img_prefix=data_root, 
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
    ),

   val=dict(
        type=dataset_type,
        classes = classes, ###########
        ann_file=data_root + 'stratified_kfold/basic_v2/cv_val_3.json', ###########
        img_prefix=data_root, ###########
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        classes = classes, ###########
        ann_file=data_root + 'test.json', ###########
        img_prefix=data_root, ###########
        pipeline=test_pipeline))




# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=2,
#     train=dict(
#         type="MultiImageMixDataset",
#         dataset=dict(
#         type=dataset_type,
#         classes = classes, ###########
#         ann_file=data_root + 'stratified_kfold/basic_v2/cv_train_3.json', ###########
#         img_prefix=data_root,  ###########
#         pipeline=[dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True)],
#         filter_empty_gt=False,
#         ),
        
#         pipeline=train_pipeline),
    
#     val=dict(
#         type=dataset_type,
#         classes = classes, ###########
#         ann_file=data_root + 'stratified_kfold/basic_v2/cv_val_3.json', ###########
#         img_prefix=data_root, ###########
#         pipeline=val_pipeline),
#     test=dict(
#         type=dataset_type,
#         classes = classes, ###########
#         ann_file=data_root + 'test.json', ###########
#         img_prefix=data_root, ###########
#         pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', classwise=True)
