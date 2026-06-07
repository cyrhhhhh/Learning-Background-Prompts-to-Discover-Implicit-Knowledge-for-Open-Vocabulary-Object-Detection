import os
import os.path as osp

# dataset settings for this workspace
_base_ = 'mmdet::_base_/datasets/coco_detection.py'
dataset_type = 'CocoDataset'
project_root = os.environ.get('LBP_PROJECT_ROOT', '')
if not project_root:
    _cwd = os.getcwd()
    project_root = _cwd if osp.basename(_cwd).lower() != 'ovdet' else osp.abspath(osp.join(_cwd, '..'))

data_root = osp.join(project_root, 'data', 'raw', 'coco')
ann_root = osp.join(project_root, 'data', 'processed', 'ov_coco', 'annotations')
compat_ann_root = osp.join(ann_root, 'compat_wusize')
file_client_args = dict(backend='disk')
branch_field = ['det_batch', 'kd_batch']

det_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiBranch',
         branch_field=branch_field,
         det_batch=dict(type='PackDetInputs'))
]

ovd_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiBranch',
         branch_field=branch_field,
         kd_batch=dict(type='PackDetInputs'))
]

det_dataset = dict(
    type='CocoDataset',
    data_root='',
    ann_file=osp.join(compat_ann_root, 'instances_train2017_base.json'),
    data_prefix=dict(img=osp.join(data_root, 'train2017')),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=det_pipeline)

ovd_dataset = dict(
    type='CocoDataset',
    data_root='',
    ann_file=osp.join(compat_ann_root, 'instances_train2017_base.json'),
    data_prefix=dict(img=osp.join(data_root, 'train2017')),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=ovd_pipeline
)

batch_split = [2, 2]
train_dataloader = dict(
    batch_size=sum(batch_split),
    num_workers=sum(batch_split),
    persistent_workers=True,
    sampler=dict(type='CustomGroupMultiSourceSampler',
                 batch_size=sum(batch_split),
                 source_ratio=batch_split),
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[det_dataset, ovd_dataset])
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    persistent_workers=False,
    dataset=dict(
        data_root='',
        ann_file=osp.join(data_root, 'annotations', 'instances_val2017.json'),
        data_prefix=dict(img=osp.join(data_root, 'val2017')),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=osp.join(compat_ann_root, 'instances_val2017_base.json'),
        metric='bbox',
        prefix='Base',
        format_only=False),
    dict(
        type='CocoMetric',
        ann_file=osp.join(compat_ann_root, 'instances_val2017_novel.json'),
        metric='bbox',
        prefix='Novel',
        format_only=False)
]
test_evaluator = val_evaluator
