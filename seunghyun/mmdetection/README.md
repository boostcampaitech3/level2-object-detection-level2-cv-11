## Train

1. `/configs/default_runtime.py`에서 WandbLoggerHook 바꿔주기
2. 아래 명령어 실행

```
sh train.sh
```

- `configs`: 학습에 사용할 config 위치
- `--work-dir`: 모델 및 로그 저장 위치
- `--resume-from`: 이어서 학습시킬 때 해당 모델의 위치


## Inference

1.  학습 후 아래 명령어 실행

```
sh inference.sh
```

- `--config`: 추론에 사용될 모델 config
- `--epoch`: 추론에 사용될 에폭
- `work_dir`: 모델이 저장된 위치 및 submission 파일이 저장될 위치