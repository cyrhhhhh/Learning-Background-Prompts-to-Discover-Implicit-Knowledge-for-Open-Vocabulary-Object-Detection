"""BCP: Background Category-specific Prompt (CVPR 2024 LBP)."""
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmengine.runner.amp import autocast
from mmdet.registry import MODELS
from ovdet.methods.builder import OVD
from ovdet.methods.baron.baron_base import BaronBase
from .utils import BackgroundFeatureQueue, BGProposalSelector, extract_clip_image_features_for_boxes, kmeans_clustering


@OVD.register_module()
class BCP(nn.Module):
    def __init__(self, num_bg_clusters=6, queue_size=2048, cluster_interval=500, start_iter=5000,
                 bg_objectness_thr=0.5, bg_weight=0.1, num_words=6, word_dim=512,
                 words_drop_ratio=0.5, cls_temp=50.0, use_attn12_output=False, use_pos_embed=True,
                 sampling_cfg=None, clip_data_preprocessor=None, gamma=0.1, **kwargs):
        super().__init__()
        self.num_bg_clusters = num_bg_clusters
        self.queue_size = queue_size
        self.cluster_interval = cluster_interval
        self.start_iter = start_iter
        self.bg_objectness_thr = bg_objectness_thr
        self.bg_weight = bg_weight
        self.num_words = num_words
        self.word_dim = word_dim
        self.words_drop_ratio = words_drop_ratio
        self.cls_temp = cls_temp
        self.use_attn12_output = use_attn12_output
        self.use_pos_embed = use_pos_embed
        self.gamma = gamma
        self.bg_selector = BGProposalSelector(bg_objectness_thr=bg_objectness_thr)
        self.bg_feat_queue = BackgroundFeatureQueue(queue_size=queue_size, feat_dim=word_dim)
        if clip_data_preprocessor is not None:
            self.clip_data_preprocessor = MODELS.build(clip_data_preprocessor)
        else:
            self.clip_data_preprocessor = None
        self.bg_prompts = nn.Parameter(torch.randn(num_bg_clusters, num_words, word_dim) * 0.02)
        if use_pos_embed:
            from ovdet.methods.baron.utils import SinePositionalEncoding
            self.pos_embed = SinePositionalEncoding(num_feats=128, num_words=num_words, word_dims=word_dim)
        else:
            self.pos_embed = None
        self.register_buffer('iter_counter', torch.tensor(0, dtype=torch.long))
        self.register_buffer('cluster_centers', torch.zeros(num_bg_clusters, word_dim))
        self.register_buffer('num_cached', torch.tensor(0, dtype=torch.long))

    @property
    def device(self):
        return self.bg_prompts.device

    def _get_positions_for_clusters(self):
        k, device = self.num_bg_clusters, self.device
        positions = torch.linspace(0.1, 0.9, k, device=device)
        cx = torch.cos(positions * 3.14159) * 0.4 + 0.5
        cy = torch.sin(positions * 3.14159) * 0.4 + 0.5
        return torch.stack([cx - 0.05, cy - 0.05, cx + 0.05, cy + 0.05], dim=1)

    def _compute_bg_cls_logits(self, region_embeddings, clip_model):
        if region_embeddings.shape[0] == 0:
            return region_embeddings.new_zeros(0, self.num_bg_clusters)
        text_encoder = clip_model.text_encoder
        use_amp = region_embeddings.is_cuda
        with autocast(enabled=use_amp):
            bg_prompts = self.bg_prompts.unsqueeze(0)
            if self.pos_embed is not None and self.use_pos_embed:
                positions = self._get_positions_for_clusters()
                pos_embeds = self.pos_embed(positions.to(self.device))
                bg_prompts_enc = bg_prompts[0] + pos_embeds
            else:
                bg_prompts_enc = bg_prompts[0]
            word_masks = self._drop_word(bg_prompts_enc)
            pseudo_text, end_token_ids = text_encoder.prepare_pseudo_text_tensor(bg_prompts_enc, word_masks)
            if self.use_attn12_output:
                bg_text_features, _, _ = text_encoder.encode_pseudo_text_endk(
                    pseudo_text, end_token_ids, text_pe=True, stepk=12, normalize=True)
            else:
                bg_text_features = text_encoder.encode_pseudo_text(
                    pseudo_text, end_token_ids, text_pe=True, normalize=True)
            region_masks = self._drop_word(region_embeddings)
            region_text, region_end_ids = text_encoder.prepare_pseudo_text_tensor(region_embeddings, region_masks)
            if self.use_attn12_output:
                region_features, _, _ = text_encoder.encode_pseudo_text_endk(
                    region_text, region_end_ids, text_pe=True, stepk=12, normalize=True)
            else:
                region_features = text_encoder.encode_pseudo_text(
                    region_text, region_end_ids, text_pe=True, normalize=True)
            logits = self.cls_temp * region_features @ bg_text_features.T
        return logits

    def _drop_word(self, word_embeddings):
        p = self.words_drop_ratio
        num_preds, num_words, _ = word_embeddings.shape
        mask = F.dropout(word_embeddings.new_ones(num_preds, num_words), p=p, training=self.training)
        start_end_mask = torch.ones_like(mask[:, :1])
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0
        mask[mask > 0.0] = 1.0
        return torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

    @torch.no_grad()
    def _cluster_bg_features(self):
        features = self.bg_feat_queue.get_all()
        if features.shape[0] < self.num_bg_clusters:
            return
        centers, labels = kmeans_clustering(features, k=self.num_bg_clusters, max_iter=100)
        self.cluster_centers = centers.to(self.device)
        self.num_cached = torch.tensor(features.shape[0], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _extract_bg_clip_features(self, images, bg_proposals, clip_model):
        if len(bg_proposals) == 0:
            return torch.empty(0, self.word_dim, device=self.device)
        return extract_clip_image_features_for_boxes(
            images=images, boxes=bg_proposals.bboxes,
            clip_model=clip_model, clip_data_preprocessor=self.clip_data_preprocessor)

    def sample(self, rpn_results, batch_data_sample):
        rpn_results.set_metainfo(batch_data_sample.metainfo)
        gt_bboxes = batch_data_sample.gt_instances.bboxes
        img_h, img_w = rpn_results.img_shape
        image_box = rpn_results.bboxes.new_tensor([0, 0, img_w - 1, img_h - 1])
        nmsed = BaronBase.preprocess_proposals(
            rpn_results, image_box[None], shape_ratio_thr=0.25, area_ratio_thr=0.01,
            objectness_thr=self.bg_objectness_thr, nms_thr=0.5)
        bg_proposals = self.bg_selector.filter_bg_proposals(nmsed, gt_bboxes)
        if len(bg_proposals) == 0:
            k = min(10, len(rpn_results))
            _, indices = rpn_results.scores.topk(k, largest=False)
            bg_proposals = rpn_results[indices]
        return bg_proposals

    def get_losses(self, region_embeddings, sampling_results, clip_model, images):
        """Compute L'_bcp loss according to Eq.(7).

        Paper logic:
        - Standard Lbcp: push sum of probs over CO U {cbg} to 1.0
        - Relaxed Lrlx: when pbg_o < gamma, uniformly push each category
        """
        self.iter_counter += 1
        losses = {}
        if self.iter_counter < self.start_iter:
            return losses
        if clip_model is not None:
            for res, img in zip(sampling_results, images):
                if len(res) > 0:
                    clip_feats = self._extract_bg_clip_features(img.unsqueeze(0), res, clip_model)
                    self.bg_feat_queue.enqueue(clip_feats)
        if self.iter_counter % self.cluster_interval == 0:
            self._cluster_bg_features()
        if region_embeddings.shape[0] == 0 or not self.bg_feat_queue.is_valid():
            return losses
        bg_logits = self._compute_bg_cls_logits(region_embeddings, clip_model)
        if bg_logits.shape[0] == 0:
            return losses
        with torch.no_grad():
            text_encoder = clip_model.text_encoder
            region_masks = self._drop_word(region_embeddings)
            region_text, region_end_ids = text_encoder.prepare_pseudo_text_tensor(
                region_embeddings, region_masks)
            if self.use_attn12_output:
                region_features, _, _ = text_encoder.encode_pseudo_text_endk(
                    region_text, region_end_ids, text_pe=True, stepk=12, normalize=True)
            else:
                region_features = text_encoder.encode_pseudo_text(
                    region_text, region_end_ids, text_pe=True, normalize=True)
            sim = region_features @ self.cluster_centers.T
            bg_labels = sim.argmax(dim=-1)

        # Compute pbg_o = sum of probs over CO U {cbg}
        bg_probs = bg_logits.softmax(dim=-1)
        pbg_o = bg_probs.sum(dim=-1)  # sum over all bg cluster probs

        # Set gamma threshold as in paper Eq.(7)
        gamma = 0.1

        # Standard Lbcp: maximize sum of bg cluster probs
        loss_bcp = F.cross_entropy(bg_logits, bg_labels)

        # Relaxed Lrlx: when pbg_o < gamma, uniformly push each bg category
        # Eq.(6): Lrlx = (1/|N|) * sum(1/noa * sum(-log p(c|x)))
        relaxed_mask = pbg_o < gamma
        if relaxed_mask.any():
            bg_logits_relaxed = bg_logits[relaxed_mask]
            if bg_logits_relaxed.shape[0] > 0:
                # noa = num_bg_clusters + 1 (for cbg)
                noa = self.num_bg_clusters + 1
                # For each proposal, compute average -log p over all bg categories
                log_probs = bg_logits_relaxed.log_softmax(dim=-1)
                # Only take the sum over CO U {cbg} categories (all logits here are bg)
                loss_rlx_per_proposal = -log_probs.sum(dim=-1) / noa
                loss_rlx = loss_rlx_per_proposal.mean()
            else:
                loss_rlx = torch.tensor(0.0, device=self.device)
        else:
            loss_rlx = torch.tensor(0.0, device=self.device)

        # Eq.(7): L'_bcp = Lbcp if pbg_o >= gamma, else Lrlx
        if relaxed_mask.any():
            loss_bcp_final = loss_rlx
        else:
            loss_bcp_final = loss_bcp

        losses['loss_bcp'] = loss_bcp_final * self.bg_weight
        return losses