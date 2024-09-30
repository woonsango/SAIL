data_root = '/home/mila/q/qian.yang/scratch/segmentation_datasets/pascalvoc20/VOCdevkit/VOC2012'
dataset_type = 'PascalVOC20Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(interval=2, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    device='cuda',
    gmp_groups=512,
    head_weights_path=
    '/network/scratch/l/le.zhang/light_align/logs/cc5mfrom12mraw_gtendinoL_bs_32768_lion_mean_lr_1e-5_star7L_d1024_scale10_negbias10/checkpoints/epoch_35.pt',
    linear_type='star',
    name_path=
    '/home/mila/q/qian.yang/Light_Align/evaluation/ClearCLIP/configs/cls_voc20.txt',
    precision='fp32',
    save_dir='/home/mila/q/qian.yang/scratch/tmp',
    target_dimension=1024,
    text_model_name='Alibaba-NLP/gte-large-en-v1.5',
    type='VLContrastModelSegmentation',
    use_gmp=False,
    vision_model_name='facebook/dinov2-large')
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root=
        '/home/mila/q/qian.yang/scratch/segmentation_datasets/pascalvoc20/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                448,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOC20Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        448,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './segmentation'
