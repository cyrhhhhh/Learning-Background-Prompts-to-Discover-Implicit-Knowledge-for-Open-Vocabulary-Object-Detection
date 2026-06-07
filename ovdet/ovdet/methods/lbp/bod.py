"""BOD: Background Object Discovery (CVPR 2024 LBP).

Paper Eq.(10): Lbod = (1/|NBp|) * sum(-log p(c=yo(x)|x))
                   + lambda_bg * (1/|NBn|) * sum(-log sum p(c|x))
                        for c in Ca U {cbg}

Key logic:
1. Extract CLIP features from background proposals
2. Compute similarity with BCP cluster centers
3. Select top-k proposals per cluster as positive samples NBp
4. Remaining proposals become negative samples NBn
5. Positive loss: CE on assigned background category
6. Negative loss: push down sum of probs over (Ca U {cbg})
"""
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from ovdet.methods.builder import OVD
from .utils import extract_clip_image_features_for_boxes


@OVD.register_module()
class BOD(nn.Module):
    def __init__(self, discovery_thr=0.7, bod_weight=0.1, topk_per_cluster=5,
                 min_proposals_for_discovery=10, single_temp=50.0,
                 neg_bg_weight=0.05,
                 clip_data_preprocessor=None, **kwargs):
        super().__init__()
        self.discovery_thr = discovery_thr
        self.bod_weight = bod_weight
        self.topk_per_cluster = topk_per_cluster
        self.min_proposals_for_discovery = min_proposals_for_discovery
        self.single_temp = single_temp
        self.neg_bg_weight = neg_bg_weight  # lambda_bg in Eq.(10)
        if clip_data_preprocessor is not None:
            self.clip_data_preprocessor = MODELS.build(clip_data_preprocessor)
        else:
            self.clip_data_preprocessor = None
        self.bcp_module = None

    @property
    def device(self):
        return next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')

    def _get_cluster_centers(self):
        """Get cluster centers from BCP module."""
        if self.bcp_module is not None:
            cc = getattr(self.bcp_module, 'cluster_centers', None)
            if cc is not None and cc.shape[0] > 0:
                return cc
        return torch.empty(0, 512, device=self.device)

    @torch.no_grad()
    def _discover_positives_and_negatives(self, bg_proposals, images, clip_model):
        """Discover positive (NBp) and negative (NBn) background proposals.

        Per paper Section 3.2.2:
        - For each estimated background category in CO, select top-k proposals
          as positive samples NBp
        - All remaining background proposals are collected as NBn
        """
        cluster_centers = self._get_cluster_centers()
        if cluster_centers.shape[0] == 0:
            return None, None, None
        all_features = []
        for proposals, img in zip(bg_proposals, images):
            if len(proposals) == 0:
                continue
            feats = extract_clip_image_features_for_boxes(
                images=img.unsqueeze(0), boxes=proposals.bboxes,
                clip_model=clip_model,
                clip_data_preprocessor=self.clip_data_preprocessor)
            if feats.shape[0] == 0:
                continue
            all_features.append(feats)
        if len(all_features) == 0:
            return None, None, None
        all_features = torch.cat(all_features, dim=0)
        num_proposals = all_features.shape[0]
        sim = all_features @ cluster_centers.T
        best_sim, best_labels = sim.max(dim=-1)
        num_clusters = cluster_centers.shape[0]
        pos_mask = torch.zeros(num_proposals, dtype=torch.bool, device=all_features.device)
        for c in range(num_clusters):
            cluster_mask = (best_labels == c)
            cluster_indices = torch.where(cluster_mask)[0]
            if cluster_indices.shape[0] == 0:
                continue
            cluster_sims = best_sim[cluster_indices]
            k = min(self.topk_per_cluster, cluster_indices.shape[0])
            _, topk_in_cluster = cluster_sims.topk(k)
            topk_global = cluster_indices[topk_in_cluster]
            pos_mask[topk_global] = True
        above_thr = best_sim > self.discovery_thr
        pos_mask = pos_mask | above_thr
        if pos_mask.sum() < self.min_proposals_for_discovery:
            k = min(self.topk_per_cluster * num_clusters, num_proposals)
            topk_k = min(max(k, self.min_proposals_for_discovery), num_proposals)
            _, indices = best_sim.topk(topk_k)
            pos_mask = torch.zeros(num_proposals, dtype=torch.bool, device=all_features.device)
            pos_mask[indices] = True
        pos_features = all_features[pos_mask]
        pos_labels = best_labels[pos_mask]
        neg_features = all_features[~pos_mask]
        return pos_features, pos_labels, neg_features

    def get_losses(self, region_embeddings, sampling_results, clip_model, images):
        """Compute Lbod loss according to Eq.(10)."""
        losses = {}
        cluster_centers = self._get_cluster_centers()
        if cluster_centers.shape[0] == 0:
            return losses
        pos_features, pos_labels, neg_features = self._discover_positives_and_negatives(
            sampling_results, images, clip_model)
        if pos_features is None or pos_features.shape[0] < 1:
            return losses
        sim = self.single_temp * (pos_features @ cluster_centers.T)
        loss_bod_pos = F.cross_entropy(sim, pos_labels)
        if neg_features is not None and neg_features.shape[0] >= 1:
            neg_sim = self.single_temp * (neg_features @ cluster_centers.T)
            neg_prob_sum = neg_sim.softmax(dim=-1).sum(dim=-1)
            neg_prob_sum = neg_prob_sum.clamp(min=1e-12, max=1.0)
            loss_bod_neg = -neg_prob_sum.log().mean()
        else:
            loss_bod_neg = torch.tensor(0.0, device=self.device)
        loss_bod = loss_bod_pos + self.neg_bg_weight * loss_bod_neg
        if torch.isfinite(loss_bod):
            losses['loss_bod'] = loss_bod * self.bod_weight
        return losses