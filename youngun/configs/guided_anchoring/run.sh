cd /opt/ml/detection/baseline/mmdetection/
python tools/train.py ../../../_boost_/youngun/configs/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco.py \
--work-dir /opt/ml/save/ga-faster-resneXt-1024