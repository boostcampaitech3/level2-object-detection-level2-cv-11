python ./inference.py \
--config ./faster_rcnn_x101_64x4d_fpn_1x_coco.py \
--epoch epoch_30 \
--work_dir /opt/ml/save/faster-rcnn-resneXt-1024


# # Single
# python tools/inference.py \
# --config /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/configs/collections/faster_rcnn_r50_fpn_1x_coco_single_target.py \
# --epoch epoch_12 \
# --work_dir /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/faster_rcnn_r50_fpn_single_target

