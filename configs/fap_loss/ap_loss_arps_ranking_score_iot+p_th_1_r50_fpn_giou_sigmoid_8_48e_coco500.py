_base_ = [
    '../_base_/datasets/coco500_detection_augm.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet',
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
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='APLossPAAHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        covariance_type='diag',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=4,
            scales_per_octave=2,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2),
    ),)
        #loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.1,
        neg_iou_thr=0.1,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
# train_cfg = dict(
#     assigner=dict(type='ATSSAssigner', topk=9),
#     allowed_border=-1,
#     pos_weight=-1,
#     debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.15,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)

# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(step=[32, 44])
total_epochs = 48