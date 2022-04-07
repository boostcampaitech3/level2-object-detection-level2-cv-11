# UniverseNet 

[공식 깃헙](https://github.com/shinya7y/UniverseNet) 참고 

UniverseNets are state-of-the-art detectors for universal-scale object detection. 
Please refer to our paper for details. https://arxiv.org/abs/2103.14027

UniverseNet을 실행하려면 mmdetection과 같은 동작방식임에도 따로 깃헙에서 클론해 라이브러리를 설치해야 한다. 
conda 가상환경이 필요하므로 주의. 

### 해당 도메인에 적합하게 바꾸면서 유의한 점 
- boostcamp 데이터셋은 COCO데이터셋과 다른 classes number (80->10)
- 기본적으로 configs/universenet/models/universenet101_gfl.py 을 상속받음


### Train 
`cd /opt/ml/UniverseNet` 에서 시작 

```
python tools/train.py configs/_univ/univ_base_3.py 
```
모델 config가 들어있는 univ.py 위치를 정확하게 지정해주기 


### Test

```
python tools/test.py  configs/_univ/univ_base_3.py  work_dirs/univ_base_3/best_bbox_mAP_epoch_*.pth --out output.pickle 
```
config 파일, pth 파일을 차례대로 파라미터로 넘겨준다. 
test하면서 나온 output이 output.pickle에 저장된다.


### Inference
```
python pkl_inference.py --pkl output.pickle --csv submission.csv
```

베이스코드 inference.ipynb를 참고해 원하는 출력방식에 맞게 submission file를 만든다. 


### Resume 
학습을 이어가고 싶을 때 
```
load_from = '/opt/ml/UniverseNet/work_dirs/univ_base_3/epoch_1.pth' # Default=None
resume_from = '/opt/ml/UniverseNet/work_dirs/univ_base_3/epoch_1.pth' # Default=None
```
그 전까지 저장했던 epoch.pth파일을 univ_runtime.py에서 `load_from`, `resume_from` 인자를 수정해서 weight를 받아와 학습을 이어나갈 수 있다. 

### TIP
1. `work_dirs/*/~.log` 를 보면 가장 윗부분에 `model`, `data`, `optimizer` 등 모델이 실제로 사용한 파라미터를 확인할 수 있다. \
코드 자주 고치다보면 예전 파라미터가 궁금해질 때가 있는데 이부분을 참고하면 좋다. 


![스크린샷 2022-04-04 오전 11 48 45](https://user-images.githubusercontent.com/68208055/161465837-bcfcd612-f96c-46d1-8f53-3c9e659da187.png)

2. batch size는 되도록 큰게 도움이 된다. batch size 8로 하다가 멈추고 batch size 16으로 바꿨는데 초반만 비교해도 batch size 16이 좋은 모습을 보이고 있다. 


### 학습 과정 
grouped OneOf by '[]'

| id | Summary | Aug | va | test mAP |
| --- | --- | --- | --- | --- |
| 1 | 기본 baseline  | multi_scale_resize, [VerticalFlip, HorizontalFlip], [GaussNoise, GaussianBlur, Blur], [RandomGamma, HueSaturationValue, ChannelDropout, ChannelShuffle, RGBShift], [ShiftScaleRotate, RandomRotate90] | fold 1  | 0.6128 |
| 2 | 기본 baseline + changed aug | multi_scale_resize, [Flip, RandomRotate90], [RandomBrightnessContrast, HueSaturationValue, GaussNoise], [Blur, GaussianBlur, MedianBlur, MotionBlur] | fold 3 |  |
| 3 | SGD Optimizer + changed aug | multi_scale_resize, [HorizontalFlip, VerticalFlip, Affine, ShiftScaleRotate], [RGBShift, ToGray, HueSaturationValue, RandomBrightnessContrast], [Blur, MedianBlur], [CLAHE, Sharpen], Emboss | fold 4  | 0.6231 |
| 4 | changed baseline(Res2Net101 → ResNet50) | same as #3  | fold 5 | bad graph  |
| 5 | 기본 baseline + changed aug  | RandomSizedBBoxSafeCrop(erosion_rate 0.3) + same as #3 | fold 5  | little bad graph |
| 6 | 기본 baseline but save best mAP_s | same as #3 | fold 3 | best epoch at 2 |
|  |  |  |  |  |


![스크린샷 2022-04-04 오후 12 03 11](https://user-images.githubusercontent.com/68208055/161467050-ca8639d4-2c4e-4f1d-b7de-30bba4eedc1f.png)


## Deformable DETR 

pytorch 1.11 이상에서만 실행할 수 있는 것 같다. 
서버 할당된 걸 그대로 쓰면 파이토치 때문에 오류가 날 수 있다. 

### Deformable DETR 학습 과정  
| id | Summary | Aug | va | test mAP |
| --- | --- | --- | --- | --- |
| 1 | 기본 baseline + AdamW | RandomSizedBBoxSafeCrop(erosion_rate 0.3) + best aug | fold 3 | 0.394 |
| 2 | 기본 baseline + AdamW | best aug | fold 1 | 0.464 |
| 3 | 기본 baseline + SGD | best aug | fold 2 | 0 |

SGD는 이상하게 하나도 안 보였다. 