## 매우 작은 bounding box 제거


- under_01: bbox_ratio < 0.1 제거
- under_03: bbox_ratio < 0.3 제거
- under_05: bbox_ratio < 0.5 제거

## 제거되는 객체 수

|  	| **under_01** 	| **under_03** 	| **under_05** 	|
|:---:	|:---:	|:---:	|:---:	|
| **General trash** 	| 109 	| 683 	| 1035 	|
| **Paper** 	| 101 	| 694 	| 1081 	|
| **Paper pack** 	| 6 	| 90 	| 166 	|
| **Metal** 	| 20 	| 118 	| 186 	|
| **Glass** 	| 6 	| 67 	| 107 	|
| **Plastic** 	| 44 	| 310 	| 541 	|
| **Styrofoam** 	| 6 	| 52 	| 107 	|
| **Plastic bag** 	| 39 	| 299 	| 511 	|
| **Battery** 	| 0 	| 3 	| 7 	|
| **Clothing** 	| 2 	| 7 	| 16 	|