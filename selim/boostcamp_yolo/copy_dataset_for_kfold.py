#-*- coding: utf-8 -*-
import os
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from sklearn.model_selection import KFold
from collections import Counter

import json
import pandas as pd

def copy_files(files, prefix):
    os.makedirs(prefix, exist_ok=True)
    for instance in tqdm(files):
        file_name = os.path.basename(instance)
        save_path = os.path.join(prefix, file_name)
        shutil.copy(instance, save_path)


if __name__ == '__main__':
    full_images = np.array(natsorted(glob('../dataset/train/*.jpg')))
    full_labels = np.array(natsorted(glob('../dataset/yolo_labels/train/*.txt')))
    
    for i in range(1, 6):
        annotations = f'/opt/ml/detection/boostcamp_yolo/fold/cv_train_{i}.json'
        with open(annotations) as f:
            data = json.load(f)

        train_dict = data.copy()
        type_ = annotations.split('/')[-1].split('.')[0]
        # print(type_)
        # print(len(data['images']))
        idxs = [i['id'] for i in data['images']]
        train_images = full_images[idxs]
        train_labels = full_labels[idxs]
        # print(train_images)
        copy_files(train_images, prefix=f'../dataset/yolo/{i}fold/train/images/')
        copy_files(train_labels, prefix=f'../dataset/yolo/{i}fold/train/labels/')
        
        
        annotations = f'/opt/ml/detection/dataset/fold/cv_val_{i}.json'
        with open(annotations) as f:
            data = json.load(f)

        train_dict = data.copy()
        type_ = annotations.split('/')[-1].split('.')[0]
        # print(type_)
        # print(len(data['images']))
        idxs = [i['id'] for i in data['images']]
        val_images = full_images[idxs]
        val_labels = full_labels[idxs]
        # print(train_images)
        copy_files(val_images, prefix=f'../dataset/yolo/{i}fold/valid/images/')
        copy_files(val_labels, prefix=f'../dataset/yolo/{i}fold/valid/labels/')