import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return A.Compose([
        A.Resize(height=512, width=512),
        A.OneOf([
            A.Flip(p=1.0),
            A.RandomRotate90(p=1.0),
            ], p=1.0),
        
        A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=1.5, sat_shift_limit=2.5, val_shift_limit=1.0, p=0.5),
        A.OneOf([
            A.Blur(p=1.0),
            A.GaussianBlur(p=1.0),
            A.MotionBlur(p=1.0),
        ], p=0.3),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        A.Resize(height=512, width=512),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
