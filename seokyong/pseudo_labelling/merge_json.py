import os
import re
import sys
import json
import argparse
import collections
from typing_extensions import final

import numpy as np
import pandas as pd
import PIL.Image as Image
import albumentations as A

from tqdm import tqdm


def read_json(json_dir: str)->pd.DataFrame:
    with open(json_dir) as json_file:
        source_anns = json.load(json_file)

    # source_annotations = source_anns['annotations']
    # source_images = source_anns['images']

    return source_anns


def save_anno(final_json, output_path):
     # save processed_anno
    with open(output_path, 'w') as outfile:
        json.dump(final_json, outfile, indent=2)

    print(f"file saved : {output_path}")

train_json = read_json('/opt/ml/detection/dataset/Pseudo/cv_train_1.json')
sudo_json = read_json('/opt/ml/detection/dataset/pseudo.json')

anno_id_start, image_id_start = train_json['annotations'][-1]['id'], train_json['images'][-1]['id']

image_id = dict()
for img in sudo_json['images']:
    image_id_start += 1
    origin_id = img['id']
    image_id[origin_id] = image_id_start

    img['id'] = image_id_start

for anno in sudo_json['annotations']:
    anno_id_start += 1
    anno['image_id'] = image_id[anno['image_id']]
    anno['id'] = anno_id_start

train_json['annotations'] += sudo_json['annotations']
train_json['images'] += sudo_json['images']
print('train + pseudo length :', len(train_json['images']))
save_anno(train_json, '/opt/ml/detection/dataset/Pseudo/train_pseudo.json')