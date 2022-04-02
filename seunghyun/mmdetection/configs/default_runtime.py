checkpoint_config = dict(max_keep_ckpts=5, interval=1) #### 최근 5개 모델만 저장

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',
                interval=1000,
                init_kwargs=dict(
                project= 'two-stage-model',
                entity = 'canvas11',
                name = 'LEE_DetectoRS_cascade_rcnn_r50_1x_Mosaic_finetune' ######
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
