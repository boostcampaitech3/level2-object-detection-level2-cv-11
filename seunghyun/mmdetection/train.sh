python tools/train.py \
 configs/collections/detectors_cascade_rcnn_r50_1x_coco.py \
 --work-dir work_dirs/detectors_cascade_rcnn_r50_1x \
#  --resume-from /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/work_dirs/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/epoch_24.pth


# ### Training Single Target Model ###
# python tools/train.py \
#  /opt/ml/detection/SEUNGHYUN_WORKSPACE/my_mmdetection/configs/collections/cascade_rcnn_swin-s_single_target.py \
#  --work-dir work_dirs/cascade_rcnn_swin-s_single_target \
