# _boostcamp/dataset_train.py

# dataset settings
dataset_type="CocoDataset"
data_root = '/opt/ml/detection/dataset/'
classes =['General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']

img_norm_cfg = dict(
    mean=[0,0,0], std=[255.,255.,255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True), #, with_mask=True),
    dict(type='Resize', img_scale=[(1024,1024), (768,768), (512,512)], keep_ratio=True,\
                multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='CutOut', n_holes=10, cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.15,0.15)]),
    dict(type='BrightnessTransform', level=2, prob=0.3), 
    dict(type="RandomAffine"),
    dict(type="RandomShift", filter_thr_px=64),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']), #, 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'fold/cv_train_5.json',
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'fold/cv_val_5.json',
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'], save_best='bbox_mAP_50', interval=1)
