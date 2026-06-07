# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.models.roi_heads import StandardRoIHead
from mmengine.structures import InstanceData
from ovdet.methods.builder import OVD


@MODELS.register_module()
class OVDStandardRoIHead(StandardRoIHead):
    def __init__(self, clip_cfg=None, ovd_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_cfg is None:
            self.clip = None
        else:
            self.clip = MODELS.build(clip_cfg)
        if ovd_cfg is not None:
            ovd_modules = {}
            for k, v in ovd_cfg.items():
                module = OVD.build(v)
                setattr(self, k, module)
                ovd_modules[k] = module
            # Establish cross-references between BCP, BOD, and IPR
            bcp_mod = ovd_modules.get('bcp', None)
            bod_mod = ovd_modules.get('bod', None)
            ipr_mod = ovd_modules.get('ipr', None)
            if bcp_mod is not None:
                bcp_mod.num_bg_clusters = getattr(bcp_mod, 'num_bg_clusters', 6)
            if bod_mod is not None:
                bod_mod.bcp_module = bcp_mod
            if ipr_mod is not None:
                ipr_mod.bcp_module = bcp_mod
            # Inject IPR into bbox_head for inference rectification
            if ipr_mod is not None and hasattr(self, 'bbox_head'):
                self.bbox_head.ipr_module = ipr_mod

    def _bbox_forward(self, x, rois):
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, self.clip)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def run_ovd(self, x, batch_data_samples, rpn_results_list, ovd_name, batch_inputs,
                *args, **kwargs):
        ovd_method = getattr(self, ovd_name)

        # Some OVD modules (e.g. BOD) do not have their own sample() method,
        # they reuse the sampling results from BCP (which samples background proposals).
        # Those modules delegate sample() to the BCP module.
        if hasattr(ovd_method, 'sample'):
            sampling_results_list = list(map(ovd_method.sample, rpn_results_list, batch_data_samples))
        elif hasattr(ovd_method, 'bcp_module') and ovd_method.bcp_module is not None:
            sampling_results_list = list(map(
                lambda r, d: ovd_method.bcp_module.sample(r, d),
                rpn_results_list, batch_data_samples
            ))
        else:
            sampling_results_list = [InstanceData() for _ in batch_data_samples]

        if isinstance(sampling_results_list[0], InstanceData):
            rois = bbox2roi([res.bboxes for res in sampling_results_list])
        else:
            sampling_results_list_ = []
            bboxes = []
            for sampling_results in sampling_results_list:
                bboxes.append(torch.cat([res.bboxes for res in sampling_results]))
                sampling_results_list_ += sampling_results
            rois = bbox2roi(bboxes)
            sampling_results_list = sampling_results_list_

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.vision_to_language(bbox_feats)
        # For baron, region embeddings are pseudo words

        return ovd_method.get_losses(region_embeddings, sampling_results_list, self.clip, batch_inputs)
