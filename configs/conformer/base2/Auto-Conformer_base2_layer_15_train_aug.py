find_unused_parameters=True
manipulate_arch=True
work_dir = './Auto-Conformer_base2_layer_15_train_aug'
repeated_aug=True

# dataset config
dataset_type = 'ImageNet'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], 
                    std=[58.395, 57.12, 57.375], 
                    to_rgb=True)
policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(
        type='Rotate',
        interpolation='bicubic',
        magnitude_key='angle',
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='horizontal'),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='vertical'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='horizontal'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='vertical')
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='RandAugment', 
        policies=policies,
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5), 
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=10,
    train=dict(
        type='ImageNet',
        data_prefix='/cluster_public_data/imagenet/origin/train',
        ann_file='/running_package/train_labeled.txt',
        pipeline=train_pipeline),
    val=dict(
        type='ImageNet',
        data_prefix='/running_package/val',
        ann_file='/running_package/val_labeled.txt',
        pipeline=test_pipeline),
    test=dict(
        type='ImageNet',
        data_prefix='/running_package/val',
        ann_file='/running_package/val_labeled.txt',
        pipeline=test_pipeline))

# default_runtime config
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# schedule config
evaluation = dict(interval=500, metric='accuracy')
optimizer = dict(type='AdamW',
                 lr=0.001,#0.001,
                 weight_decay=0.05,
                 eps=1e-8, # new added
                 betas=(0.9, 0.999),
                 paramwise_cfg = dict(
                 norm_decay_mult=0.0,
                 bias_decay_mult=0.0,
                 custom_keys={
                     '.backbone.cls_token': dict(decay_mult=0.0),
                     '.rel_pos_embed_k': dict(decay_mult=0.0),
                     '.rel_pos_embed_v': dict(decay_mult=0.0)}))
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=20,#5,
    warmup_ratio=1e-3,#1e-3#1e-6,
    warmup_by_epoch=True)
# runner = dict(type='EpochBasedRunnerDistill', max_epochs=500)
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(interval=10)

# supernet config
model = dict(
    type='ElasticeImageClassifierConformer',
    backbone=dict(
        type='ElasticConformer',
        embed_dim=624,#
        num_heads=10,#
        mlp_ratio=4,#
        depth=15,#
        stem_width=80,#
        body_width=[80,160,320],#
        body_depth=[5,5,4],#
        patch_size=16,
        in_chans=3,
        ),
    neck=None,
    head=dict(
        type='ElasticConformerClsHead',
        num_classes=1000,
        in_channels=624,#
        channel_ratio=320,#
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1, 5)),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)])
)
# sample config
train_sampler = dict(
    type='concat',
    model_samplers=[
        dict(
            type='anchor',
            anchors=[
                dict({
                    'name': 'MAX',
                    'arch.backbone.stem.width': 80,
                    'arch.backbone.body.depth': [5,5,4],
                    'arch.backbone.body.block.convblock.width': [80,160,320],
                    'arch.backbone.body.block.embed_dim.width': 624,
                    'arch.backbone.body.block.transblock.MHA.num_heads.num_heads': [10,10,10],
                    'arch.backbone.body.block.transblock.FFN.feedforward_channels.feedforward_channels': [40,40,40]
                }),
                dict({
                    'name': 'MIN',
                    'arch.backbone.stem.width': 48,
                    'arch.backbone.body.depth': [4,4,3],
                    'arch.backbone.body.block.convblock.width': [48,96,192],
                    'arch.backbone.body.block.embed_dim.width': 528,
                    'arch.backbone.body.block.transblock.MHA.num_heads.num_heads': [8,8,8],
                    'arch.backbone.body.block.transblock.FFN.feedforward_channels.feedforward_channels': [35,35,35]
                }),
                dict({
                    'name': 'new-base',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.depth': [4,4,3],
                    'arch.backbone.body.block.convblock.width': [64,128,256],
                    'arch.backbone.body.block.embed_dim.width': 576,
                    'arch.backbone.body.block.transblock.MHA.num_heads.num_heads': [9,9,9],
                    'arch.backbone.body.block.transblock.FFN.feedforward_channels.feedforward_channels': [40,40,40]
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
                        start=48,
                        end=80,
                        step=16),
                    dict(
                        type='range',
                        key='arch.backbone.body.depth',
                        start=[4,4,3],
                        end=[5,5,4],
                        step=[1,1,1]),
                    dict(
                        type='range',
                        key='arch.backbone.body.block.convblock.width',
                        start=[48,96,192],
                        end=[80,160,320],
                        step=[16,32,64]),
                    dict(
                        type='range',
                        key='arch.backbone.body.block.embed_dim.width',
                        start=528,
                        end=624,
                        step=48),
                    dict(
                        type='range',
                        key='arch.backbone.body.block.transblock.MHA.num_heads.num_heads',
                        start=[8,8,8],
                        end=[10,10,10],
                        step=[1,1,1]),
                    dict(
                        type='range',
                        key='arch.backbone.body.block.transblock.FFN.feedforward_channels.feedforward_channels',
                        start=[35,35,35],
                        end=[40,40,40],
                        step=[5,5,5]),
                ]))
    ])
val_sampler = dict(
    type='anchor',
    anchors=[
            dict({
                'name': 'new-base',
                'arch.backbone.stem.width': 64,
                'arch.backbone.body.depth': [4,4,3],
                'arch.backbone.body.block.convblock.width': [64,128,256],
                'arch.backbone.body.block.embed_dim.width': 576,
                'arch.backbone.body.block.transblock.MHA.num_heads.num_heads': [9,9,9],
                'arch.backbone.body.block.transblock.FFN.feedforward_channels.feedforward_channels': [40,40,40]
            })
    ])