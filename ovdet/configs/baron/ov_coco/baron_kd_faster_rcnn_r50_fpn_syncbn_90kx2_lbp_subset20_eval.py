_base_ = './baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_mini50.py'

import os
import os.path as osp

project_root = os.environ.get('LBP_PROJECT_ROOT', '')
if not project_root:
    _cwd = os.getcwd()
    project_root = _cwd if osp.basename(_cwd).lower() != 'ovdet' else osp.abspath(osp.join(_cwd, '..'))

data_root = osp.join(project_root, 'data', 'raw', 'coco')
subset_ann_root = osp.join(project_root, 'outputs', 'ovdet_subset_compat')

small_test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(512, 320), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file=osp.join(subset_ann_root, 'instances_val2017_subset20.json'),
        data_prefix=dict(img=osp.join(data_root, 'val2017')),
        test_mode=True,
        pipeline=small_test_pipeline))

test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=osp.join(subset_ann_root, 'instances_val2017_base_subset20.json'),
        metric='bbox',
        prefix='Base',
        format_only=False),
    dict(
        type='CocoMetric',
        ann_file=osp.join(subset_ann_root, 'instances_val2017_novel_subset20.json'),
        metric='bbox',
        prefix='Novel',
        format_only=False)
]
