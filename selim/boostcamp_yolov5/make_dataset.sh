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