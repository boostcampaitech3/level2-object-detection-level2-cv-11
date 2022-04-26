# 🎨 [Pstage] CV 11조 CanVas 

<p align="center">
<img width="1078" alt="스크린샷 2022-04-27 오전 12 02 43" src="https://user-images.githubusercontent.com/63924704/165331044-f08b1dd6-30a5-4bd9-b575-f54fc3ff1034.png">
</p>

- 대회 기간 : 2022.03.21 ~ 2022.04.08
- 목적 : 재활용 품목 분류를 위한 Object Detection

---

## 🔎 Overview

 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제가 나오고 있다. 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나이다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문이다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결하고자 한다.

---

## 💾  데이터셋

<p align="center">
<img width="600" alt="스크린샷 2022-04-27 오전 12 02 43" src="https://user-images.githubusercontent.com/68208055/164621090-2ac83869-d6b6-4b6a-bde4-fe5275252d83.png">

</p>

- 전체 이미지 개수 : 9754장 (train 4883 장, test 4871 장)
- 10개 클래스 : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)

---

## 🧑‍🤝‍🧑 멤버
 
| 김영운 | 이승현 | 임서현 | 전성휴 | 허석용 |  
| :-: | :-: | :-: | :-: | :-: |  
|[Github](https://github.com/Cronople) | [Github](https://github.com/sseunghyuns) | [Github](https://github.com/seohl16) | [Github](https://github.com/shhommychon) | [Github](https://github.com/HeoSeokYong)

- `김영운` : Faster R-CNN(ResNeXt), Guided-Anchoring-Faster 설계, Mosaic augmentation 실험
- `이승현` : EfficientDet, DetectoRS 모델 실험, Input size & augmentation 실험, anchor scale 실험
- `임서현` : HTC(ResNeXt), YOLOv5, UniverseNet, Deformable DETR 설계 및 실험, Ensemble
- `전성휴` : CenterNet, TridentNet, TOOD 등 모델 실험, 앙상블 종류 실험
- `허석용` : HTC, Cascade, Faster R-CNN(SwinT) 설계 및 실험, TTA, Pseudo labeling 기법 설계

---

## 🛰️ 프로젝트 수행 결과 

<p align="center">
<img width="715" alt="2" src="https://user-images.githubusercontent.com/63924704/165333567-65991267-3919-449f-91da-8340ed89bc2c.png">
</p>

<p align="center">
<img width="1112" alt="스크린샷 2022-04-27 오전 12 15 05" src="https://user-images.githubusercontent.com/63924704/165333856-c11877b3-82ad-4053-a9fc-d721a8ba960e.png">
</p>

## 다양한 모델 실험 
1-stage detector
- EfficientDet → 5 fold ensemble → 0.5217
- YOLOv5 → 0.4543 → 5fold & pseudo → 0.6071
- UniverseNet → 0.6128 ​​→ 5fold → 0.6466

2-stage detector
- Guided-Anchroing-Faster-ResNeXt → 0.548 →pseudo→ 0.582
- DetectoRS → 0.553
- Swin-tiny Faster R-CNN → 0.516
- Swin-small Cascade R-CNN → 0.597
- Swin-base Hybrid Task Cascade R-CNN → 0.6393
- Swin-large Cascade R-CNN → 0.629



## Reference
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [UniverseNet](https://github.com/shinya7y/UniverseNet)
