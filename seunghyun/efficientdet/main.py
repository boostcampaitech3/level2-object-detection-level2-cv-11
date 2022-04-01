import wandb
import torch
from args import Args
from util import *
from augmentation import *
from trainer import train
from model import get_efficientdet

from dataloader import MaskDataset
from torch.utils.data import DataLoader

def main(args, wandb):
    data_dir = '/opt/ml/detection/dataset/'
        
    train_dataset = MaskDataset(args['TRAIN_ANN'], data_dir, get_train_transform())
    train_data_loader = DataLoader(train_dataset, batch_size=args['BATCH_SIZE'], shuffle=True, num_workers=2, collate_fn=collate_fn)

    valid_dataset = MaskDataset(args['VAL_ANN'], data_dir, get_valid_transform())
    valid_data_loader = DataLoader(valid_dataset, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=2, drop_last=False,  collate_fn=collate_fn)
    
    gt = get_gt(args['VAL_ANN'])
    
    model = get_efficientdet(num_classes=10, img_scale=args['IMG_SCALE'], model_type=args['MODEL_TYPE'])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args['LEARNING_RATE'])

    train(args, model, optimizer, train_data_loader, valid_data_loader, gt, wandb)
    
    
if __name__ == '__main__':
    args = Args().params
    print("\n=========Training Info=========\n"
            "Model type: {}".format(args['MODEL_TYPE']), "\n",
            "Img scale: {}".format(args['IMG_SCALE']), "\n",
            "Batch size: {}".format(args['BATCH_SIZE']), "\n",
            "Learning rate: {}".format(args['LEARNING_RATE']), "\n",
            "Fold: {}".format(args['SAVE_DIR']), "\n", 
            "===============================")
    
    wandb.init(project="one-stage-model", entity="canvas11", name = args['WANDB_NAME'])

    main(args, wandb)
