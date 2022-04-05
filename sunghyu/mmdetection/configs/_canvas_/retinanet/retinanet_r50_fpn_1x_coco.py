_base_ = [
    '_canvas_base_/models/retinanet_r50_fpn.py',
    '_canvas_base_/datasets/coco_detection.py',
    '_canvas_base_/schedules/sgd_cosann.py', '_canvas_base_/default_runtime.py'
]
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
