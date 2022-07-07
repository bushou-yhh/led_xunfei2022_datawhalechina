# dataset settings
dataset_type = 'LED'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
classes= ['good', 'bad']
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict( 
            type=dataset_type,
            data_prefix='data/led/trainval',
            ann_file='data/led/trainval/train.txt',
            classes=classes,
            pipeline=train_pipeline)
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/led/trainval',
        ann_file='data/led/trainval/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/led/trainval',
        ann_file='data/led/trainval/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ))
evaluation = dict(interval=1, metric='accuracy')
