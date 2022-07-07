_base_ = [
    './_base_/models/hrnet/hrnet-w18.py',
    './_base_/datasets/led_bs32_pil_resize.py',
    './_base_/default_runtime.py'
]


model = dict(
    type='ImageClassifier',
    backbone=dict(type='HRNet', arch='w18'),
    neck=[
        dict(type='HRFuseScales', in_channels=(18, 36, 72, 144)),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,2 ),
    ))


# optimizer
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=50)

evaluation = dict(interval=1, metric=[ 'accuracy', 'precision', 'recall', 'f1_score', 'support'])