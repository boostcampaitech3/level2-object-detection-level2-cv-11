_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_change_iou_softnms.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]),
    # CE Loss -> Focal Loss
    # roi_head=dict(
    #     bbox_head=[dict(
    #         loss_cls=dict(
    #           type='FocalLoss',
    #           gamma=2.0,
    #           alpha=0.25,
    #           use_sigmoid=True, 
    #           loss_weight=1.0)
    #     ),
    #     dict(
    #         loss_cls=dict(
    #           type='FocalLoss',
    #           gamma=2.0,
    #           alpha=0.25,
    #           use_sigmoid=True, 
    #           loss_weight=1.0)
    #     ),
    #     dict(
    #         loss_cls=dict(
    #           type='FocalLoss',
    #           gamma=2.0,
    #           alpha=0.25,
    #           use_sigmoid=True, 
    #           loss_weight=1.0)
    #     )]
    # )
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                           (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                           (736, 1024), (768, 1024), (1024, 1024)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1024), (500, 1024), (600, 1024)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(512, 512),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1024), (512, 1024), (544, 1024),
                                     (576, 1024), (608, 1024), (640, 1024),
                                     (672, 1024), (704, 1024), (736, 1024),
                                     (768, 1024), (1024, 1024)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(
    policy='CosineAnnealing',
    warmup_iters=488, # 1 epoch
    min_lr=1e-06, 
    # step=[27, 33]
    )
runner = dict(max_epochs=36)
