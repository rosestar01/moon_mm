# Refer to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification
# ----------------------------
# -[x] auto_augment='imagenet'
# -[x] batch_size=128 (per gpu)
# -[x] epochs=600
# -[x] opt='rmsprop'
#     -[x] lr=0.064
#     -[x] eps=0.0316
#     -[x] alpha=0.9
#     -[x] weight_decay=1e-05
#     -[x] momentum=0.9
# -[x] lr_gamma=0.973
# -[x] lr_step_size=2
# -[x] nproc_per_node=8
# -[x] random_erase=0.2
# -[x] workers=16 (workers_per_gpu)
# - modify: RandomErasing use RE-M instead of RE-0

_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]



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

policies = [
    [
        dict(type='Posterize', bits=4, prob=0.4),
        dict(type='Rotate', angle=30., prob=0.6)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
    [
        dict(type='Posterize', bits=5, prob=0.6),
        dict(type='Posterize', bits=5, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 6, prob=0.6),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Posterize', bits=6, prob=0.8),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='Rotate', angle=10., prob=0.2),
        dict(type='Solarize', thr=256 / 9, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.6),
        dict(type='Posterize', bits=5, prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
        dict(type='ColorTransform', magnitude=0., prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.0),
     dict(type='Equalize', prob=0.8)],
    [dict(type='Invert', prob=0.6),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
        dict(type='ColorTransform', magnitude=0.2, prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0.8, prob=0.8),
        dict(type='Solarize', thr=256 / 9 * 2, prob=0.8)
    ],
    [
        dict(type='Sharpness', magnitude=0.7, prob=0.4),
        dict(type='Invert', prob=0.6)
    ],
    [
        dict(
            type='Shear',
            magnitude=0.3 / 9 * 5,
            prob=0.6,
            direction='horizontal'),
        dict(type='Equalize', prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    [dict(type='Invert', prob=0.6),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
]



# data = dict(
#     samples_per_gpu=128,
#     workers_per_gpu=4,
#     train=dict(pipeline=train_pipeline))
dataset_type = 'CustomDataset'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_prefix='E:/python-project/other/mmclassification/data/flower_imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='E:/python-project/other/mmclassification/data/flower_imagenet/val',
        ann_file='E:/python-project/other/mmclassification/data/flower_imagenet/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='E:/python-project/other/mmclassification/data/flower_imagenet/val',
        ann_file='E:/python-project/other/mmclassification/data/flower_imagenet/val.txt',
        pipeline=test_pipeline))




model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeSt',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=5,
            reduction='mean',
            loss_weight=1.0),
        topk=(1),
        cal_acc=True))
train_cfg = dict(mixup=dict(alpha=0.2, num_classes=5))



#  evaluation = dict(interval=1, metric='accuracy')
evaluation = dict(interval=1, metric_options={'topk': (1, )})

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=2,
    hooks=[
        # dict(type='TextLoggerHook'),

        dict(type='TensorboardLoggerHook')
    ])
# learning policy


optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.01,  weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
