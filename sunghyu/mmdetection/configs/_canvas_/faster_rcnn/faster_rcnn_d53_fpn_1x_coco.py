_base_ = [
    '_canvas_base_/models/faster_rcnn_r50_fpn.py',
    '_canvas_base_/datasets/coco_detection.py',
    '_canvas_base_/schedules/schedule_1x.py', '_canvas_base_/default_runtime.py'
]

model = dict(
    backbone=dict(_delete_=True, type='CSPDarknet', deepen_factor=0.67, widen_factor=0.75, out_indices=(0, 1, 2, 3)),
    neck=dict(in_channels=[48, 96, 192, 384]),
)