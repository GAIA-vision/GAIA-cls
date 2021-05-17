model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop_cls', size=32, padding=4),
    dict(type='RandomFlip_cls', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize_cls', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect_cls', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize_cls', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect_cls', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/cifar10',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type, data_prefix='data/cifar10', pipeline=test_pipeline),
    test=dict(
        type=dataset_type, data_prefix='data/cifar10', pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

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
work_dir = '/mnt/diske/qing_chang/GAIA/workdirs/gaia-seg-modelsample-class-test'

'''
stem_width_range = dict(
    key='arch.backbone.stem.width', start=64, end=64, step=16)
body_width_range = dict(
    key='arch.backbone.body.width',
    start=[80, 160, 320, 640],
    end=[80, 160, 320, 640],
    step=[16, 32, 64, 128],
    ascending=True)
body_depth_range = dict(
    key='arch.backbone.body.depth',
    start=[4, 6, 29, 4],
    end=[4, 6, 29, 4],
    step=[1, 1, 1, 1])
MAX = dict({
    'name': 'MAX',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [80, 160, 320, 640],
    'arch.backbone.body.depth': [4, 6, 29, 4]
})
MIN = dict({
    'name': 'MIN',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [80, 160, 320, 640],
    'arch.backbone.body.depth': [4, 6, 29, 4]
})
R50 = dict({
    'name': 'R50',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 6, 3]
})
R77 = dict({
    'name': 'R77',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 15, 3]
})
R101 = dict({
    'name': 'R101',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 23, 3]
})
train_sampler = dict(
    type='concat',
    model_samplers=[
        dict(
            type='anchor',
            anchors=[
                dict({
                    'name': 'MAX',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [64, 128, 256, 512],
                    'arch.backbone.body.depth': [3, 4, 23, 3]
                }),
                dict({
                    'name': 'MIN',
                    'arch.backbone.stem.width': 16,
                    'arch.backbone.body.width': [48, 96, 192, 384],
                    'arch.backbone.body.depth': [2, 2, 5, 2]
                })
            ]),
        dict(
            type='repeat',
            times=3,
            model_sampler=dict(
                type='composite',
                model_samplers=[
                    dict(
                        type='range',
                        key='arch.backbone.stem.width',
                        start=32,
                        end=64,
                        step=16),
                    dict(
                        type='range',
                        key='arch.backbone.body.width',
                        start=[48, 96, 192, 384],
                        end=[64, 128, 256, 512],
                        step=[16, 32, 64, 128],
                        ascending=True),
                    dict(
                        type='range',
                        key='arch.backbone.body.depth',
                        start=[2, 2, 5, 2],
                        end=[3, 4, 23, 3],
                        step=[1, 1, 1, 1])
                ]))
    ])
val_sampler = dict(
    type='anchor',
    anchors=[
        dict({
            'name': 'R50',
            'arch.backbone.stem.width': 64,
            'arch.backbone.body.width': [64, 128, 256, 512],
            'arch.backbone.body.depth': [3, 4, 6, 3]
        }),
        dict({
            'name': 'R77',
            'arch.backbone.stem.width': 64,
            'arch.backbone.body.width': [64, 128, 256, 512],
            'arch.backbone.body.depth': [3, 4, 15, 3]
        }),
        dict({
            'name': 'R101',
            'arch.backbone.stem.width': 64,
            'arch.backbone.body.width': [64, 128, 256, 512],
            'arch.backbone.body.depth': [3, 4, 23, 3]
        })
    ])
'''
