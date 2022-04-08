_base_ = [
    '../models/cascade_rcnn_r50_fpn_anchor_scale.py', ############## # Detector + Backbone 교체 완료
    '../datasets/coco_detection.py', ##############
    '../schedules/schedule_3x.py', '../default_runtime.py'
]

###### Swin Transformer-Large ######
# backbone
pretrained = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"  # noqa
)
model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        # pretrain_img_size=384, ### ? 뭐지 이건
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48], ######
        window_size=12,
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
    neck=dict(in_channels=[192, 384, 768, 1536]), ########
)