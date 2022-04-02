# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
ann_root = '/opt/ml/_boost_/stratified_kfold/basic_v2/'
classes =['General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']

img_scale=(1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
    type='OneOf',
    transforms=[
        dict(type='Flip',p=1.0),
        dict(type='RandomRotate90',p=1.0)
    ],
    p=0.5),
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

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    
    dict(type='Mosaic', img_scale=img_scale),
    #dict(
    #    type='RandomAffine',
    #    scaling_ratio_range=(0.1, 2),
    #    border=(-img_scale[0] // 2, -img_scale[1] // 2)),
        # The image will be enlarged by 4 times after Mosaic processing,
        # so we use affine transformation to restore the image size.
    
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

train_dataset = dict(
    # _delete_ = True, # remove unnecessary Settings
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=ann_root + 'cv_train_3.json',
        classes=classes,
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
    ),
    pipeline=train_pipeline
    )


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

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'cv_val_3.json',
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(metric='bbox', save_best='bbox_mAP_50', interval=1, classwise=True)        
#evaluation = dict(interval=1, metric='bbox')
