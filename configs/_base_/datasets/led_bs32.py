_base_ = ['./pipelines/rand_aug.py']
# dataset settings
dataset_type = 'LED'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
classes= ['good', 'bad']
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict( 
            type=dataset_type,
            data_prefix='data/led/',
            ann_file='data/led/train.txt',
            classes=classes,
            pipeline=train_pipeline)
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/led/',
        ann_file='data/led/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/led',
        ann_file='data/led/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ))

evaluation = dict(interval=1, metric=['f1_score', 'accuracy'])
