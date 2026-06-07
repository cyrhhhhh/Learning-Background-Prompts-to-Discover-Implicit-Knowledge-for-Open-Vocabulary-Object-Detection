import os
import os.path as osp

_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn_syncbn.py',
    '../../_base_/datasets/coco_ovd_kd_ms_lbp.py',
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
compat_ann_root = osp.join(project_root, 'data', 'processed', 'ov_coco', 'annotations', 'compat_wusize')
class_weight = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                1, 0, 1, 1, 1, 1, 0, 0, 0, 1] + [0.7]

reg_layer = [
    dict(type='Linear', in_features=1024, out_features=1024),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=1024, out_features=4)
]

clip_cfg = dict(
    type='CLIP',
    image_encoder=dict(
        type='CLIPViT',
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        init_cfg=dict(
            type='Pretrained',
            prefix='visual',
            checkpoint=osp.join(checkpoint_root, 'clip_vitb32.pth'))
    ),
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

clip_data_preprocessor=dict(
    type='ImgDataPreprocessor',
    mean=[(122.7709383 - 123.675) / 58.395,
          (116.7460125 - 116.28) / 57.12,
          (104.09373615 - 103.53) / 57.375],
    std=[68.5005327 / 58.395,
         66.6321579 / 57.12,
         70.32316305 / 57.375])

sampling_cfg=dict(shape_ratio_thr=0.25,
                  area_ratio_thr=0.01,
                  objectness_thr=0.85,
                  nms_thr=0.1,
                  topk=300,
                  max_groups=3,
                  max_permutations=2,
                  alpha=3.0,
                  cut_off_thr=0.3,
                  base_probability=0.3,
                  interval=-0.1,
                  )

# Original BARON KD configuration
ovd_baron_cfg = dict(type='BaronKD',
                     boxes_cache=dict(
                         json_path=osp.join(compat_ann_root, 'instances_train2017_base.json'),
                         start_iter=20000,
                     ),
                     use_gt=True,
                     bag_weight=1.0,
                     single_weight=0.1,
                     use_attn_mask=False,
                     bag_temp=30.0,
                     single_temp=50.0,
                     clip_data_preprocessor=clip_data_preprocessor,
                     num_words=6,
                     word_dim=512,
                     words_drop_ratio=0.5,
                     queue_cfg=dict(names=['clip_text_features', 'clip_image_features',
                                           'clip_word_features', 'clip_patch_features'],
                                    lengths=[1024] * 4,
                                    emb_dim=512,
                                    id_length=1),
                     sampling_cfg=sampling_cfg,
                     )

# BCP: Background Category-specific Prompt
ovd_bcp_cfg = dict(type='BCP',
                   num_bg_clusters=6,
                   queue_size=2048,
                   cluster_interval=500,
                   start_iter=5000,
                   bg_objectness_thr=0.5,
                   bg_weight=0.1,
                   num_words=6,
                   word_dim=512,
                   words_drop_ratio=0.5,
                   cls_temp=50.0,
                   use_attn12_output=False,
                   use_pos_embed=True,
                   clip_data_preprocessor=clip_data_preprocessor,
                   )

# BOD: Background Object Discovery
ovd_bod_cfg = dict(type='BOD',
                   discovery_thr=0.7,
                   bod_weight=0.1,
                   topk_per_cluster=5,
                   min_proposals_for_discovery=10,
                   single_temp=50.0,
                   clip_data_preprocessor=clip_data_preprocessor,
                   )

model = dict(
    type='OVDTwoStageDetector',
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        _delete_=True,
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32),
    ),
    rpn_head=dict(
        type='DetachRPNHead',
        anchor_generator=dict(
            scale_major=False,
        )
    ),
    # kd_batch runs three OVD modules: baron_kd (BARON region-text KD),
    # bcp (Background Category-specific Prompt), and bod (Background Object Discovery)
    batch2ovd=dict(kd_batch=['baron_kd', 'bcp', 'bod']),
    roi_head=dict(
        type='OVDStandardRoIHead',
        clip_cfg=clip_cfg,
        ovd_cfg=dict(baron_kd=ovd_baron_cfg,
                     bcp=ovd_bcp_cfg,
                     bod=ovd_bod_cfg),
        bbox_head=dict(
            type='BaronShared4Conv1FCBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=None,
            cls_temp=50.0,
            num_words=6,
            cls_embeddings_path=osp.join(metadata_root, 'coco_clip_hand_craft_attn12.npy'),
            bg_embedding='learn',
            use_attn12_output=True,
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight),
        ),
    ),
    test_cfg=dict(
        rcnn=dict(score_thr=0.01)
    )
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.002 * 2, momentum=0.9, weight_decay=0.000025),
)
load_from = osp.join(checkpoint_root, 'res50_fpn_soco_star_400.pth')
