
checkpoint_config = dict(interval=8, max_keep_ckpts=3)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),  
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='one-stage-model',
                name='LIM-deformable-DETR-SGD',
                entity='canvas11')
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from ='https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
resume_from = None
workflow = [('train', 1)]

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
