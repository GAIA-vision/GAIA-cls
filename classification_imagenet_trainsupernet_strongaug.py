# ref a strong augment setting from
# https://rwightman.github.io/pytorch-image-models/training_hparam_examples/#resnet50-with-jsd-loss-and-randaugment-clean-2x-ra-augs-7904-top-1-9439-top-5
# resnet50
# 可以查下mmcv.configFromfile的那个API是否支持config里面有if语句。
# rand-m9-mstd0.5-inc1
'''s
dict(type='RandAugment',
     policies=[
         dict(type='AutoContrast',prob=0.5),
         dict(type='Equalize',prob=0.5),
         dict(type='Invert',prob=0.5),
         dict(type='Rotate',prob=0.5),
         dict(type='Posterize',prob=0.5),
         dict(type='Solarize',prob=0.5),
         dict(type='SolarizeAdd',prob=0.5),
         dict(type='ColorTransform',prob=0.5),
         dict(type='Contrast',prob=0.5),
         dict(type='Brightness',prob=0.5),
         dict(type='Sharpness',prob=0.5),
         dict(type='Shear',prob=0.5,direction='horizontal'),
         dict(type='Shear',prob=0.5,direction='vertical'),
         dict(type='Translate',prob=0.5,direction='horizontal'),
         dict(type='Translate',prob=0.5,direction='vertical')
     ],
     num_policies=2,
     magnitude_level = 27
     magnitude_std = 0.5
)'''
aug_num_split = 3
split_bn = False
model = dict(
    type='DynamicImageClassifier',
    augmix_used = True,
    aug_split_num = 3,
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
        loss=dict(type='JsdCrossEntropyLoss', num_splits=aug_num_split, loss_weight=1.0),
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
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='AugMix',aug_splits=aug_num_split),
    # --aa rand-m9-mstd0.5-inc1  mmcls本身就提供AutoAugment，看看是直接复用mmcls的还是复用这个强baseline的
    # dict(type='AutoAugment_augmix',aa='rand-m9-mstd0.5-inc1',mode='AugMix')
    # --remode pixel --reprob 0.6 
    # dict(type='RandomErasing',remode='pixel',reprob=0.6)
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
    samples_per_gpu=64, # -b 64
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
            dict(type='AugMix',aug_splits=3),
            # --remode pixel --reprob 0.6 
            dict(type='RandomErasing',mode='pixel',probability=0.6,num_splits=3),
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

# --lr 0.05  momentum 和 weight decay如何设置还得check一下。
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# todo
# --sched cosine 
lr_config = dict(policy='step', step=[30, 60, 90])

# --epochs 200 这个可能还有点问题，因为那个强baseline打印的时候，epoch数目不是200 是210 好像是有默认的warmup
# mmcv支持设置warmup 不过因为强baseline里面的不是显示设置的，还得check下它的warmup具体是怎样的。
# ref: https://github.com/open-mmlab/mmcv/blob/e728608ac9/mmcv/runner/hooks/lr_updater.py(Line 25)
runner = dict(type='EpochBasedRunner', max_epochs=200)
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
