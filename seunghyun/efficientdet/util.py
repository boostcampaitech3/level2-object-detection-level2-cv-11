from tqdm import tqdm
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

def collate_fn(batch):
    return tuple(zip(*batch))

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def get_gt(GT_JSON):
    gt = []
    coco = COCO(GT_JSON)
    for image_id in coco.getImgIds():

        image_info = coco.loadImgs(image_id)[0]
        annotation_id = coco.getAnnIds(imgIds=image_info['id'])
        annotation_info_list = coco.loadAnns(annotation_id)

        file_name = image_info['file_name']

        for annotation in annotation_info_list:
            gt.append([file_name, annotation['category_id'],
                       float(annotation['bbox'][0]),
                       float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                       float(annotation['bbox'][1]),
                       (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
            
    return gt


def get_submission(outputs, valid_annotation, score_threshold=0.1, valid=False):
    prediction_strings = []
    file_names = []
    coco = COCO(valid_annotation)

    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds()[i])[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0]*2) + ' ' + str(
                    box[1]*2) + ' ' + str(box[2]*2) + ' ' + str(box[3]*2) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    
    if valid:
        new_pred = []

        file_names = submission['image_id'].values.tolist()
        bboxes = submission['PredictionString'].values.tolist()

        for i, bbox in enumerate(bboxes):
            if isinstance(bbox, float):
                print(f'{file_names[i]} empty box')

        for file_name, bbox in tqdm(zip(file_names, bboxes)):
            boxes = np.array(str(bbox).split(' '))

            if len(boxes) % 6 == 1:
                boxes = boxes[:-1].reshape(-1, 6)
            elif len(boxes) % 6 == 0:
                boxes = boxes.reshape(-1, 6)
            else:
                raise Exception('error', 'invalid box count')
            for box in boxes:
                new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])
        
        
        return new_pred
    
    return submission