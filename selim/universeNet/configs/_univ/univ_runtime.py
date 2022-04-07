
runner = dict(type='EpochBasedRunner', max_epochs=40)

checkpoint_config = dict(interval=5, max_keep_ckpts=3)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),  
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='one-stage-model',
                name='LIM-UniverseNet-SGD-40epoch-CLAHE',
                entity='canvas11')
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from ='https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth'
resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    save_best='bbox_mAP',
    metric=['bbox']
)