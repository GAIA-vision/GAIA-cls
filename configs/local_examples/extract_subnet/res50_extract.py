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
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

train_cfg = dict()
test_cfg = dict(mode='whole')
input_shape_cands = dict(
    key='data.input_shape',
    candidates=(480, 560, 640, 720, 800, 880, 960)
)
stem_width_range = dict(
    key='arch.backbone.stem.width',
    start=32,
    end=64,
    step=16,
)
body_width_range = dict(
    key='arch.backbone.body.width',
    start=[48, 96, 192, 384],
    end=[80, 160, 320, 640],
    step=[16, 32, 64, 128],
    ascending=True,
)
body_depth_range = dict(
    key='arch.backbone.body.depth',
    start=[2, 2, 5, 2],
    end=[4, 6, 29, 4],
    step=[1, 2, 2, 1],
)

# predefined model anchors
MAX = {
    'name': 'MAX',
    'arch.backbone.stem.width': stem_width_range['end'],
    'arch.backbone.body.width': body_width_range['end'],
    'arch.backbone.body.depth': body_depth_range['end'],
    'data.input_shape': 800,
}
MIN = {
    'name': 'MIN',
    'arch.backbone.stem.width': stem_width_range['start'],
    'arch.backbone.body.width': body_width_range['start'],
    'arch.backbone.body.depth': body_depth_range['start'],
    'data.input_shape': 800,
}
R50 = {
    'name': 'R50',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 6, 3],
    'data.input_shape': 800,
}
R77 = {
    'name': 'R77',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 15, 3],
    'data.input_shape': 800,
}
R101 = {
    'name': 'R101',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 23, 3],
    'data.input_shape': 800,
}



train_sampler = dict(
    type='anchor',
    anchors=[
        dict(
            **R50,
        ),
        #dict(
        #    **R101,
        #),
        dict(
            **R77,
        )
    ]
)


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
checkpoint_config = dict(by_epoch=False, interval=8000)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
