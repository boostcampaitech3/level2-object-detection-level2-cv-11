## 폴더 트리

‘*’ 직접 만들어야 하는 거 git에 올려놓을 예정 

‘+’ 깃 클론할거 

```
── baseline/
│   ├── mmdetection/
│   ├── faster_rcnn/
│   └── detectron2/
│
├── dataset/
│   ├── *yolo/
│   ├── *yolo_labels/
│   ├── test/
│   └── train/
│
├── +convert2Yolo/
│   ├── ...
│   └── example.py
│
│── +boostcamp_yolo/
│   ├── *make_dataset.sh
│   │── *trash.names
│   ├── *split_yolo_dataset.py
│   ├── *btcamp_0f.yaml
│   └── *for_yolov5/
│       │── *0_detectandinfer.ipynb
│       │── *3_yolo_to_coco.py
│       │── *4_makejson.py
│       │── *detect.py
│       └── *run.py
│
└── +yolov5/
    ├── data/
    ├── train.py
    ├── detect.py
```

## 0. convert2Yolo - 데이터셋 YOLO Format으로 바꾸기

YOLO는 annotation 방식이 COCO랑 달라서 json 파일을 바꿀 필요가 있었다. 

참고할만한 깃헙이 많았는데 마지막으로 참고한 깃헙은 [convert2Yolo](https://github.com/ssaru/convert2Yolo)  

`make_dataset.sh`

```python
# make_yolo_dataset.sh
pip install Pillow cycler kiwisolver numpy pandas opencv-python python-dateutil pytz six matplotlib natsort tqdm scikit-learn ensemble-boxes
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

cd /opt/ml/detection
# 1. Coco to Yolov5 format dataset
git clone https://github.com/ssaru/convert2Yolo.git
cd convert2Yolo
mkdir ../dataset/yolo_labels/
mkdir ../dataset/yolo_labels/train/

# 2. data from COCO to YOLO format
python example.py --datasets COCO --img_path ../dataset/train/ --label ../dataset/train.json --convert_output_path ../dataset/yolo_labels/ --img_type ".jpg" --manifest_path ./ --cls_list_file ../boostcamp_yolo/trash.names

cd ../boostcamp_yolo
# 3. Yolov5 train/valid split -> look at dataset/yolo/ for result
python split_yolo_dataset.py

cd ..
# 4. Yolov5 clone & Prerequisite
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cp ../boostcamp_yolo/btcamp_0f.yaml ./data/
rm detect.py 
cp ../boostcamp_yolo/for_yolov5 ./
```

제일 먼저 실행할 파일이다. 

1. pip install로 필요한 라이브러리를 설치하고 
2. train.json 등 coco 데이터를 yolo 형식으로 바꾼다. 
3. 그 다음 fold를 할 경우를 대비해 split_yolo_dataset 파일을 실행시킨다. 
4. yolov5 공식 깃을 클론하고 필요한 파일을 원하는 위치에 복사시키면 끝 

- 코드 설명
    
    `trash.names` 
    
    ```python
    General trash
    Paper
    Paper pack
    Metal
    Glass
    Plastic
    Styrofoam
    Plastic bag
    Battery
    Clothing
    ```
    
    example.py에서 요구하는 클래스 이름이 담긴 파일 
    
    `split_yolo_dataset.py`
    
    ```python
    #-*- coding: utf-8 -*-
    import os
    import shutil
    import numpy as np
    from tqdm import tqdm
    from glob import glob
    from natsort import natsorted
    from sklearn.model_selection import KFold
    
    def copy_files(files, prefix):
        os.makedirs(prefix, exist_ok=True)
        for instance in tqdm(files):
            file_name = os.path.basename(instance)
            save_path = os.path.join(prefix, file_name)
            shutil.copy(instance, save_path)
    
    if __name__ == '__main__':
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        full_images = np.array(natsorted(glob('../dataset/train/*.jpg')))
        full_labels = np.array(natsorted(glob('../dataset/yolo_labels/train/*.txt')))
    
        use_folds = [0,2,4]   # 10Fold중 0,2,4 Fold만 사용함. (학습 시간 때문에)
        for fold, (trn_idx, val_idx) in enumerate(kf.split(full_images)):
            if fold not in use_folds:
                continue
    
            val_images = full_images[val_idx]
            train_images = full_images[trn_idx]
    
            val_labels = full_labels[val_idx]
            train_labels = full_labels[trn_idx]
    
            copy_files(train_images, prefix=f'./data/yolo/{fold}fold/train/images/')
            copy_files(train_labels, prefix=f'./data/yolo/{fold}fold/train/labels/')
            copy_files(val_images, prefix=f'./data/yolo/{fold}fold/valid/images/')
            copy_files(val_labels, prefix=f'./data/yolo/{fold}fold/valid/labels/')
    ```
    
    KFold 함수로 나눈 fold directory 이미지가 물리적으로 저장된다. 
    
    `btcamp_0f.yaml`
    
    ```python
    path: ../dataset/yolo/0fold  # dataset root dir
    train: train/images  # train images (relative to 'path')
    val: valid/images  # val images (relative to 'path')
    test:  # test images (optional)
    
    # Classes
    nc: 10  # number of classes
    # names: ['01_ulcer', '02_mass', '04_lymph', '05_bleeding']  # class names
    names: ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    ```
    
    yolov5에서 요구하는 data/~.yaml 파일 
    
    참고할 이미지 위치, train, valid, number of classes, names of classes로 구성된다. 
    

## 1. Train

train.py 를 실행시킨다. 
labels will be saved in weights/{experiment name}/

`--project` pth파일을 저장할 위치 \
`--name` experiment name \
`--img` image size \
`--data` yaml file for classes and data file path \
`--weights` initial weight (ex. yolov5s yolov5m yolov5l ...)\
`--save-period` save .pt file for every x epoch\
`--workers` worker for training

```python
cd /opt/ml/detection/yolov5

mkdir weights/

#내가 돌린 세팅
python train.py --project weights/ --name=exp_with_1024 \
--img 1024 --batch 16 --epochs 30 --data data/btcamp_0f.yaml \
--weights yolov5l.pt --save-period 7 --workers 1
```

공식 깃허브 - yaml 예시 참고 

[https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)

## 2. Detect

labels 아웃풋 저장할 폴더 만들기!! 
make labels/*.txt

`--weights` train 시켰던 pth 파일 위치 \
`--project` labels 폴더를 만들 위치 \
`--name` experiment name \
`--img` image size \
`--source` test_img_directory \
`--save-txt` save results in .txt \
`--save-conf` save confidence score \
`--conf-thres` confidence threshold \
`--iou-thres` NMS IOU threshold \
`--nosave` do not save image \
`--data` yaml file 

(내 경우에는 —project results로 results라는 폴더에 넣도록 했다)

```python

mkdir results/

python detect.py --weights weights/exp_with_1024/weights/best.pt \
--project results/ --name exp_with_1024 --img 1024 \
--source ../dataset/test/ --save-txt --save-conf \
--conf-thres 0.005 --iou-thres 0.5 --nosave --data data/btcamp_0f.yaml
```
    

## 3. make submission (from YOLO to COCO)

마지막으로 yolo to coco format으로 바꿔서 submission.csv에 저장한다. 
change labels/*.txt to submission.csv 

`--dir_prefix` directory where labels is \
`--exp_name` experiment name \
`--test_img_path` test_img_directory

```python
python 3_yolo_to_coco.py --dir_prefix results/ \
--exp_name exp_with_1024 --test_img_path /opt/ml/detection/dataset/test/
```

results/exp~/submission.csv로 저장된다. 

## 4. run with fiftyone

```python
cp results/exp_with_1024/submission.csv ./
python 4_makejson.py 
python run.py --data_dir test --port 30006 --anno_dir ./csv_to_json.json
```
