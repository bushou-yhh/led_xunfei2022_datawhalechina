_base_ = [
    './_base_/models/resnet50.py', './_base_/datasets/led_bs32.py',
    './_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2,
      topk=(1,2)),
)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.25)
runner = dict(type='EpochBasedRunner', max_epochs=50)

