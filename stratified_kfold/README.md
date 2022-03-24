## basic 데이터셋

- `basic`: 신규범 캠퍼님 코드 기반 kfold split 데이터셋([링크](https://stages.ai/competitions/178/discussion/talk/post/1203))
- `basic_v2`: 남혜린 조교님 코드 기반 kfold split 데이터셋([링크](https://stages.ai/competitions/178/discussion/talk/post/1205))

</br>

## Tiny bounding box 제거 데이터셋

- `under_{num}`: 남혜린 조교님 코드 기반, tiny bbox 제거된 kfold split 데이터셋

<img width="248" alt="bbox_ratio" src="https://user-images.githubusercontent.com/63924704/159717730-7682c1ea-07ed-404b-bff7-19c1ca5009b4.png">

- `under_01`: bbox_ratio < 0.1 제거
- `under_03`: bbox_ratio < 0.3 제거
- `under_05`: bbox_ratio < 0.5 제거

## 제거되는 객체 수

| **id** 	| **Category** 	| **under_01** 	| **under_03** 	| **under_05** 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|
| 0 	| General trash 	| 109 	| 683 	| 1035 	|
| 1 	| Paper 	| 101 	| 694 	| 1081 	|
| 2 	| Paper pack 	| 6 	| 90 	| 166 	|
| 3 	| Metal 	| 20 	| 118 	| 186 	|
| 4 	| Glass 	| 6 	| 67 	| 107 	|
| 5 	| Plastic 	| 44 	| 310 	| 541 	|
| 6 	| Styrofoam 	| 6 	| 52 	| 107 	|
| 7 	| Plastic bag 	| 39 	| 299 	| 511 	|
| 8 	| Battery 	| 0 	| 3 	| 7 	|
| 9 	| Clothing 	| 2 	| 7 	| 16 	|
| - 	| Total 	| 333 	| 2323 	| 3757 	|
