## HTC (Hybrid Task Cascade) 
[공식 깃헙](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc) 참고 

semantic segmentation에 더 적합한 모델이라 htc_without_semantic 파일을 상속받음. 

### 해당 도메인에 적합하게 바꾸면서 유의한 점 
- boostcamp 데이터셋은 COCO데이터셋과 다른 classes number (80->10)
- htc_without_semantic 안에 있는 mask 관련 모듈은 주석


### Train 
`cd /opt/ml/detection/baseline/mmdetection` 에서 시작 

```
python tools/train.py configs/_boostcamp/htc_class10/htc_train.py
```
htc_train.py 위치를 정확하게 지정해주기 


### Test

```
python tools/test.py configs/_boostcamp/htc/htc_train.py work_dirs/htc_train/best_bbox_mAP_50_epoch_*.pth --out output.pickle 
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
load_from = '/opt/ml/detection/baseline/mmdetection/work_dirs/htc_train/epoch_1.pth' # Default=None
resume_from = '/opt/ml/detection/baseline/mmdetection/work_dirs/htc_train/epoch_1.pth' # Default=None
```
그 전까지 저장했던 epoch.pth파일을 default_runtime.py에서 `load_from`, `resume_from` 인자를 수정해서 weight를 받아와 학습을 이어나갈 수 있다. 

### TIP
`work_dirs/htc_train/~.log` 를 보면 가장 윗부분에 `model`, `data`, `optimizer` 등 모델이 실제로 사용한 파라미터를 확인할 수 있다. \
코드 자주 고치다보면 예전 파라미터가 궁금해질 때가 있는데 이부분을 참고하면 좋다. 
