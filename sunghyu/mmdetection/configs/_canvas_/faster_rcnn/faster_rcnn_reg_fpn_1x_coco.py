_base_ = [
    '_canvas_base_/models/faster_rcnn_r50_fpn.py',
    '_canvas_base_/datasets/coco_detection.py',
    '_canvas_base_/schedules/sgd_cosann.py', '_canvas_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True, type='RegNet', arch="regnetx_800mf", out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_800mf')
    ),
    neck=dict(in_channels=[64, 128, 288, 672], out_channels=256)
)