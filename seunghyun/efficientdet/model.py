import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

# Effdet config
# https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/config/model_config.py

def get_efficientdet(num_classes, img_scale, model_type, checkpoint_path=None):
    
    config = get_efficientdet_config(model_type)
    config.num_classes = num_classes
    config.image_size = img_scale
    config.soft_nms = False
    config.max_det_per_image = 50 ###
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    net = DetBenchTrain(net)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)
        print(f'Weight loaded from -> {checkpoint_path}')
        
    return net