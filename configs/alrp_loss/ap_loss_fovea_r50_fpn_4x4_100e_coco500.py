_base_ = [
    '../_base_/datasets/coco500_detection_augm.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FOVEA',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=5,
        add_extra_convs='on_input'),
    bbox_head=dict(
        type='APLossFoveaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        base_edge_list=[16, 32, 64, 128, 256],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.4,
        with_deform=False,
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=1000,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(step=[75, 95])
total_epochs = 100