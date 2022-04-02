import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for EfficienDet')
    parser.add_argument('--model_type', default='tf_efficientdet_d3')
    parser.add_argument('--img_scale', type=int, nargs="+", default=(512, 512), help='Resize input image')
    parser.add_argument('--val_ann', default='/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_val_3.json')
    parser.add_argument('--train_ann', default='/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_train_3.json')
    parser.add_argument('--checkpoint', default='./pretrained/checkpoint512x512/epoch_36.pth')
    parser.add_argument('--save_dir', default='./pretrained/fold1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--wandb_name', default='LEE_EfficientDet_512x512_fold3')


    parse = parser.parse_args()
    params = {
        "MODEL_TYPE": parse.model_type, 
        "IMG_SCALE": parse.img_scale, 
        "VAL_ANN": parse.val_ann,
        "CHECKPOINT": parse.checkpoint, 
        "SAVE_DIR": parse.save_dir,
        "TRAIN_ANN": parse.train_ann, 
        "BATCH_SIZE": parse.batch_size,
        "LEARNING_RATE": parse.learning_rate,
        "NUM_EPOCHS": parse.num_epochs,
        "RANDOM_SEED": parse.random_seed,
        "WANDB_NAME": parse.wandb_name
    }