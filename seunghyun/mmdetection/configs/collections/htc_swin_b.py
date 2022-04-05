_base_ = [
    '../models/htc_without_semantic_r50_fpn_1x_coco_rpn_changed.py', ############## # Detector + Backbone 교체 완료
    '../datasets/coco_detection.py', ##############
    '../schedules/schedule_3x.py', '../default_runtime.py'
]

#####
pretrained = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"  # noqa
)
model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)