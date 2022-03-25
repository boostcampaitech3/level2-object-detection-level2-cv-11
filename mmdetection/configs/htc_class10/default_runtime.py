checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #'/opt/ml/detection/baseline/mmdetection/work_dirs/htc_train/epoch_1.pth'
resume_from = None # '/opt/ml/detection/baseline/mmdetection/work_dirs/htc_train/epoch_1.pth'
workflow = [('train', 1)] #, ('val', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
