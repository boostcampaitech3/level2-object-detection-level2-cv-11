# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True), ##############
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale, ##############
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes = classes, ###########
        ann_file=data_root + 'stratified_kfold/basic_v2/cv_train_3.json', ##############
        img_prefix=data_root, ##############
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = classes, ###########
        ann_file=data_root + 'stratified_kfold/basic_v2/cv_val_3.json', ##############
        img_prefix=data_root, ##############
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = classes, ###########
        ann_file=data_root + 'test.json', ##############
        img_prefix=data_root, ##############
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox') ###?
