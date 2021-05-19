# dataset settings
# model settings
model = dict(
    type='DynamicImageClassifier',
   backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=64,
        body_depth=[3,4,23,3],
        body_width=[64,128,256,512],
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        conv_cfg=dict(type='DynConv2d'),
        norm_eval=False,
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
dataset_type = 'ImageNet'
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
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/mnt/diske/qing_chang/Data/ImageNet/ILSVRC2012_img_train',
        ann_file='/mnt/diske/qing_chang/Data/ImageNet/train_labeled.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/mnt/diske/qing_chang/Data/ImageNet/ILSVRC2012_img_val',
        ann_file='/mnt/diske/qing_chang/Data/ImageNet/val_labeled.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/mnt/diske/qing_chang/Data/ImageNet/ILSVRC2012_img_val',
        ann_file='/mnt/diske/qing_chang/Data/ImageNet/val_labeled.txt',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='accuracy')
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
  
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '/mnt/diske/qing_chang/GAIA/workdirs/gaia-clas-imagenet-test'

val_sampler = dict(
    type='anchor',
    anchors=[
        dict({
            'name': 'R101',
            'arch.backbone.stem.width': 64,
            'arch.backbone.body.width': [64, 128, 256, 512],
            'arch.backbone.body.depth': [3, 4, 23, 3]
        })
    ])
train_sampler = dict(
    type='anchor',
    anchors=[
        dict({
            'name': 'R101',
            'arch.backbone.stem.width': 64,
            'arch.backbone.body.width': [64, 128, 256, 512],
            'arch.backbone.body.depth': [3, 4, 23, 3]
        })
    ])
