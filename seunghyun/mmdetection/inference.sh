python tools/inference.py \
--config /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/swinb_htc_total_data/htc_swin_b.py \
--epoch epoch_42 \
--work_dir /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/swinb_htc_total_data \
; python tools/inference.py \
--config /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s-multiscale-aug_total_data/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
--epoch epoch_30 \
--work_dir /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s-multiscale-aug_total_data \
; python tools/inference.py \
--config /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s_total_data/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
--epoch epoch_30 \
--work_dir /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s_total_data \
; python tools/inference.py \
--config /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin_large_iou_total_data/cascade_rcnn_swin_large_iou.py \
--epoch epoch_9 \
--work_dir /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin_large_iou_total_data
