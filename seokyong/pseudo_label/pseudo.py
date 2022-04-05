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
pd.set_option('display.max_columns', None)


def csv2df(output_dir: str, confidence: float):
    submission_df = pd.read_csv(output_dir)
    idcount = 0
    stack = []
    
    for i in tqdm(range(submission_df.shape[0]), leave = False):
        im_name = submission_df.iloc[i, 2]
        annos = submission_df.iloc[i, 1]
        if type(annos) == str:
            annos = annos.split(' ')
        else: continue
        
        for box_idx in range(len(annos)//6):
            st = box_idx *6
            ed = st + 6
            #annos = ['cls', 'conf', 'X_min', 'Y_min', 'X_max', 'Y_max']
            ann = annos[st:ed]
            ann = [float(x) for x in ann]
            W = ann[-2] - ann[-4]
            H = ann[-1] - ann[-3]
            row = [im_name, idcount] + [W*H] + ann[0:-2] + [W, H]
            stack.append(row)
            idcount += 1

    df = pd.DataFrame.from_records(stack, columns=['image_id', 'bbox_id', 'area', 'cls', 'conf', 'X', 'Y', 'W', 'H'])
    # df[['area', 'cls', 'conf', 'X', 'Y', 'W', 'H']] = df[['area', 'cls', 'conf', 'X', 'Y', 'W', 'H']].astype(float)
    df['cls'] = df['cls'].astype(int)
    df = df[df['conf'] > confidence]
    return df


def read_json(json_dir: str)->pd.DataFrame:
    with open(json_dir) as json_file:
        source_anns = json.load(json_file)

    # source_annotations = source_anns['annotations']
    # source_images = source_anns['images']

    return source_anns


def df2coco(source_anns: dict, processed_anno: pd.DataFrame):

    source_anns['images'] = []
    source_anns['annotations'] = []
    regex = re.compile(r'[^0-9]')
    for image_id in processed_anno.image_id.unique():
        image_info = {
                        "width": 1024,
                        "height": 1024,
                        "file_name": str(image_id),
                        "license": 0,
                        "flickr_url": 'null',
                        "coco_url": 'null',
                        "date_captured": 'null',
                        "id": int(regex.sub(r'', image_id))
        }
        source_anns['images'].append(image_info)

    for i in tqdm(range(processed_anno.shape[0]), leave=False):
        row = processed_anno.iloc[i, :]
        anno_info = {
                        "image_id": int(regex.sub(r'', row['image_id'])),
                        "category_id": int(row['cls']),
                        "area": row['area'],
                        "bbox": [row['X'], row['Y'], row['W'], row['H']],
                        "iscrowd": 0,
                        "id": int(row['bbox_id'])
        }

        source_anns['annotations'].append(anno_info)

    return source_anns


def save_anno(final_json, output_path):
     # save processed_anno
    with open(output_path, 'w') as outfile:
        json.dump(final_json, outfile, indent=2)

    print(f"file saved : {output_path}")


def main(args):
    test_json = read_json(args.test_dir)
    inference_df = csv2df(args.inf_dir, confidence = float(args.confidence))

    save_dir = os.path.join('/'.join(args.test_dir.split('/')[:-1]), 'pseudo.json')
    final_json = df2coco(test_json, inference_df)
    save_anno(final_json, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for creating pseudo label')
    parser.add_argument('--test_dir', default = '/opt/ml/detection/dataset/test.json')
    parser.add_argument('--inf_dir', default='/opt/ml/detection/baseline/mmdetection/ensemble/06891.csv')
    parser.add_argument('--confidence', default = 0.3, help = 'i')
    # parser.add_argument('--num', default = 5, help='number of mosaic image warning: image may duplicate if mosaic num > 4 * image num(with or without constraint). use at your own risk')

    args = parser.parse_args()
    main(args)