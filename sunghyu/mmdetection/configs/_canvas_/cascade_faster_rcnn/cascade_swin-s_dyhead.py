# https://github.com/boostcampaitech3/level2-object-detection-level2-cv-11/blob/master/seunghyun/mmdetection/
# configs/collections/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py

_base_ = [
    '_canvas_base_/models/ss_cascade_rcnn.py', ############## 
    '_canvas_base_/datasets/ss_coco_detection.py', ##############
    '_canvas_base_/schedules/ss_schedule_2x.py', '_canvas_base_/ss_default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2], #####
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
    neck=[
        dict(
            type='FPN',
            in_channels=[96, 192, 384, 768], 
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)
    ]
)

fp16 = dict(loss_scale=512.) ## AMP 학습

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            interval=1000,
            init_kwargs={
                "entity": "canvas11",
                "project": "two-stage-model",
                "name": "CHON_cascade_swin-s_dyhead_24e"
            }
        )
    ]
)

seed = 2022
gpu_ids = [0]
work_dir = '/opt/ml/detection/baseline/mmdetection/work_dirs/cascade_swin-s_dyhead_24e_trash'
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
