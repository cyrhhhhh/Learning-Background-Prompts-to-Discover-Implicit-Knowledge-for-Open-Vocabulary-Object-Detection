import os
import os.path as osp

_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn_syncbn.py',
    '../../_base_/datasets/coco_ovd_base_ms_lbp.py',
    '../../_base_/schedules/schedule_90k.py',
    '../../_base_/iter_based_runtime.py'
]
project_root = os.environ.get('LBP_PROJECT_ROOT', '')
if not project_root:
    _cwd = os.getcwd()
    project_root = _cwd if osp.basename(_cwd).lower() != 'ovdet' else osp.abspath(osp.join(_cwd, '..'))

ovdet_root = osp.join(project_root, 'ovdet')
checkpoint_root = osp.join(ovdet_root, 'checkpoints')
metadata_root = osp.join(ovdet_root, 'data', 'metadata')
class_weight = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                1, 0, 1, 1, 1, 1, 0, 0, 0, 1] + [1]

reg_layer = [
    dict(type='Linear', in_features=1024, out_features=1024),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=1024, out_features=4)
]

clip_cfg = dict(
    type='CLIP',
    image_encoder=None,
    text_encoder=dict(
        type='CLIPTextEncoder',
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=osp.join(checkpoint_root, 'clip_vitb32.pth'))
    )
)

model = dict(
    type='OVDTwoStageDetector',
    rpn_head=dict(
        type='DetachRPNHead',
        anchor_generator=dict(
            scale_major=False,
        )
    ),
    roi_head=dict(
        type='OVDStandardRoIHead',
        clip_cfg=clip_cfg,
        bbox_head=dict(
            type='BaronShared4Conv1FCBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=None,
            num_words=6,
            cls_temp=50.0,
            cls_embeddings_path=osp.join(metadata_root, 'coco_clip_hand_craft_attn12.npy'),
            bg_embedding='learn',
            use_attn12_output=True,
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight),
        ),
    ),
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.02 * 2, momentum=0.9, weight_decay=0.000025),
)
load_from = osp.join(checkpoint_root, 'res50_fpn_soco_star_400.pth')
