_base_ = [
    './_base_/models/swin_transformer/tiny_224.py',
    './_base_/datasets/led_bs64_swin_224.py',
    './_base_/default_runtime.py'
]



checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=2, ),
train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=2, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=2, prob=0.5)
    ]))




paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 64 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(interval=1, metric=[ 'accuracy', 'precision', 'recall', 'f1_score', 'support'])