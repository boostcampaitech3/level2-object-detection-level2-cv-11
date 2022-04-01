python tools/inference.py \
--config /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/configs/collections/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
--epoch epoch_24 \
--work_dir /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s-multiscale-aug


# # Single
# python tools/inference.py \
# --config /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/configs/collections/faster_rcnn_r50_fpn_1x_coco_single_target.py \
# --epoch epoch_12 \
# --work_dir /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/faster_rcnn_r50_fpn_single_target

