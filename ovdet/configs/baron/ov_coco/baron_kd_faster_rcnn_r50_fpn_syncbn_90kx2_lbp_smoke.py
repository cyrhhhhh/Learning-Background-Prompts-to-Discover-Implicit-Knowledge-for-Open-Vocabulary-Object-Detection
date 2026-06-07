_base_ = './baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp.py'

import os
import os.path as osp

project_root = os.environ.get('LBP_PROJECT_ROOT', '')
if not project_root:
    _cwd = os.getcwd()
    project_root = _cwd if osp.basename(_cwd).lower() != 'ovdet' else osp.abspath(osp.join(_cwd, '..'))

data_root = osp.join(project_root, 'data', 'raw', 'coco')
ann_root = osp.join(project_root, 'data', 'processed', 'ov_coco', 'annotations')
compat_ann_root = osp.join(ann_root, 'compat_wusize')

small_det_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 320), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MultiBranch',
        branch_field=['det_batch', 'kd_batch'],
        det_batch=dict(type='PackDetInputs'))
]

small_ovd_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 320), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MultiBranch',
        branch_field=['det_batch', 'kd_batch'],
        kd_batch=dict(type='PackDetInputs'))
]

small_test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(512, 320), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

det_dataset_smoke = dict(
    type='CocoDataset',
    data_root='',
    ann_file=osp.join(compat_ann_root, 'instances_train2017_base.json'),
    data_prefix=dict(img=osp.join(data_root, 'train2017')),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=small_det_pipeline)

ovd_dataset_smoke = dict(
    type='CocoDataset',
    data_root='',
    ann_file=osp.join(compat_ann_root, 'instances_train2017_base.json'),
    data_prefix=dict(img=osp.join(data_root, 'train2017')),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=small_ovd_pipeline)

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(
        type='CustomGroupMultiSourceSampler',
        batch_size=2,
        source_ratio=[1, 1]),
    batch_sampler=None,
    dataset=dict(
        type='ConcatDataset',
        datasets=[det_dataset_smoke, ovd_dataset_smoke]))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file=osp.join(data_root, 'annotations', 'instances_val2017.json'),
        data_prefix=dict(img=osp.join(data_root, 'val2017')),
        test_mode=True,
        pipeline=small_test_pipeline))
test_dataloader = val_dataloader

train_cfg = dict(max_iters=1, val_interval=1000)
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(interval=1, max_keep_ckpts=1))
log_processor = dict(window_size=1)
