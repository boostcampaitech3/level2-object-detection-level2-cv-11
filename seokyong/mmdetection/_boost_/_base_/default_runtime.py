checkpoint_config = dict(max_keep_ckpts=3, interval=1) # dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', interval=100),
        dict(type='WandbLoggerHook',interval=100,
            init_kwargs=dict(
                project='two-stage-model',
                entity = 'canvas11',
                name = 'HEO-htc-swins-focal-softnms-aug_lee'
            ),
        )
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
seed = 2022
gpu_ids = [0]
work_dir = './work_dirs/htc_swins_aug_focal_softnms_fpn_trash'

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
