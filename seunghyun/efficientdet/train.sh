python main.py --val_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_val_1.json' \
--train_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_train_1.json' \
--save_dir './pretrained/fold1' \
--wandb_name 'LEE_EfficientDet_512x512_fold1' \
; python main.py --val_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_val_2.json' \
--train_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_train_2.json' \
--save_dir './pretrained/fold2' \
--wandb_name 'LEE_EfficientDet_512x512_fold2' \
; python main.py --val_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_val_3.json' \
--train_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_train_3.json' \
--save_dir './pretrained/fold3' \
--wandb_name 'LEE_EfficientDet_512x512_fold3' \
; python main.py --val_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_val_4.json' \
--train_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_train_4.json' \
--save_dir './pretrained/fold4' \
--wandb_name 'LEE_EfficientDet_512x512_fold4' \
; python main.py --val_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_val_5.json' \
--train_ann '/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_train_5.json' \
--save_dir './pretrained/fold5' \
--wandb_name 'LEE_EfficientDet_512x512_fold5'