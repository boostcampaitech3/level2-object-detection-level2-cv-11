python tools/train.py \
 /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s-multiscale-aug/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
 --work-dir work_dirs/cascade_rcnn_swin-s-multiscale-aug_total_data \
 --resume-from /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s-multiscale-aug/epoch_24.pth \
; python tools/train.py \
 /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
 --work-dir work_dirs/cascade_rcnn_swin-s_total_data \
 --resume-from /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s/epoch_24.pth \
; python tools/train.py \
 /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/swinb_htc/htc_swin_b.py \
 --work-dir work_dirs/swinb_htc_total_data \
 --resume-from /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/swinb_htc/epoch_36.pth \
; python tools/train.py \
 /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/configs/collections/cascade_rcnn_swin_large_iou.py \
 --work-dir work_dirs/cascade_rcnn_swin_large_iou_total_data
