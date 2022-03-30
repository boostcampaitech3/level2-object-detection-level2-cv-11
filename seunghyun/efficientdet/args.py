import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for EfficienDet')
    parser.add_argument('--model_type', default='tf_efficientdet_d3')
    parser.add_argument('--img_scale', type=int, nargs="+", default=(512, 512), help='Resize input image')
    parser.add_argument('--val_ann', default='/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_val_3.json')
    parser.add_argument('--train_ann', default='/opt/ml/detection/dataset/stratified_kfold/basic_v2/cv_train_3.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=36)
    parser.add_argument('--random_seed', type=int, default=42)


    parse = parser.parse_args()
    params = {
        "MODEL_TYPE": parse.model_type, 
        "IMG_SCALE": parse.img_scale, 
        "VAL_ANN": parse.val_ann,
        "TRAIN_ANN": parse.train_ann, 
        "BATCH_SIZE": parse.batch_size,
        "LEARNING_RATE": parse.learning_rate,
        "NUM_EPOCHS": parse.num_epochs,
        "RANDOM_SEED": parse.random_seed,
    }