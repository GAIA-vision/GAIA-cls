# support distillation
model = dict(
    type='DynamicImageClassifier',
    backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=64,
        body_depth=[3, 4, 23, 3],
        body_width=[64, 128, 256, 512],
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        conv_cfg=dict(type='DynConv2d'),
        norm_eval=False,
        norm_cfg=dict(type='DynBN', requires_grad=True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='JsdCrossEntropyLoss', num_splits=3, loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32, # -b 64
    workers_per_gpu=2,
    train=dict(
        type='ImageNet',
        data_prefix='/mnt/diske/qing_chang/Data/ImageNet/ILSVRC2012_img_train',
        ann_file='/mnt/diske/qing_chang/Data/ImageNet/train_labeled.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='ImageNet',
        data_prefix='/mnt/diske/qing_chang/Data/ImageNet/ILSVRC2012_img_val',
        ann_file='/mnt/diske/qing_chang/Data/ImageNet/val_labeled.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='ImageNet',
        data_prefix='/mnt/diske/qing_chang/Data/ImageNet/ILSVRC2012_img_val',
        ann_file='/mnt/diske/qing_chang/Data/ImageNet/val_labeled.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')


optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[30, 60, 90])

# --epochs 200 这个可能还有点问题，因为那个强baseline打印的时候，epoch数目不是200 是210 好像是有默认的warmup
# mmcv支持设置warmup 不过因为强baseline里面的不是显示设置的，还得check下它的warmup具体是怎样的。
# ref: https://github.com/open-mmlab/mmcv/blob/e728608ac9/mmcv/runner/hooks/lr_updater.py(Line 25)
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


work_dir = '/mnt/diske/qing_chang/GAIA/workdirs/gaia-cls-imagenet-train-supernet'
stem_width_range = dict(
    key='arch.backbone.stem.width', start=32, end=64, step=16)
body_width_range = dict(
    key='arch.backbone.body.width',
    start=[48, 96, 192, 384],
    end=[64, 128, 256, 512],
    step=[16, 32, 64, 128],
    ascending=True)
body_depth_range = dict(
    key='arch.backbone.body.depth',
    start=[2, 2, 5, 2],
    end=[3, 4, 23, 3],
    step=[1, 2, 2, 1])
MAX = dict({
    'name': 'MAX',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 23, 3]
})
MIN = dict({
    'name': 'MIN',
    'arch.backbone.stem.width': 32,
    'arch.backbone.body.width': [48, 96, 192, 384],
    'arch.backbone.body.depth': [2, 2, 5, 2]
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
                    'name': 'MIN',
                    'arch.backbone.stem.width': 32,
                    'arch.backbone.body.width': [48, 96, 192, 384],
                    'arch.backbone.body.depth': [2, 2, 5, 2]
                }),
                dict({
                    'name': 'R101',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [64, 128, 256, 512],
                    'arch.backbone.body.depth': [3, 4, 23, 3]
                }),
                dict({
                    'name': 'R77',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [64, 128, 256, 512],
                    'arch.backbone.body.depth': [3, 4, 15, 3]
                }),
                dict({
                    'name': 'R50',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [64, 128, 256, 512],
                    'arch.backbone.body.depth': [3, 4, 6, 3]
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
                        step=[1, 2, 2, 1])
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
gpu_ids = range(0, 4)
