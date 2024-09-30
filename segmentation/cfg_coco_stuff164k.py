data_root = '/home/mila/q/qian.yang/scratch/coco_stuff164k'
dataset_type = 'COCOStuffDataset'
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
    '/home/mila/q/qian.yang/scratch/light_align/pytorch_model.bin',
    linear_type='star',
    name_path=
    '/home/mila/q/qian.yang/Light_Align/evaluation/ClearCLIP/configs/cls_coco_stuff.txt',
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
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        data_root='/home/mila/q/qian.yang/scratch/coco_stuff164k',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                448,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='COCOStuffDataset'),
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
