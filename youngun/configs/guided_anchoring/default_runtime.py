checkpoint_config = dict(interval=3, max_keep_ckpts=3) # 최근 3개 모델 저장
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',
                interval=1000,
                init_kwargs=dict(
                project='two-stage-model',
                entity = 'canvas11',
                name = 'KIM_ga-faster-resneXt-pseudo-1024'
            ),
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
