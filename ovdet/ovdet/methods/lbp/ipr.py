"""IPR: Inference Probability Rectification (CVPR 2024 LBP).

Paper Eq.(18): 修正推理时 novel 类别概率被背景潜在类别压低的问题。

核心思想:
1. 训练阶段 BCP 估计的背景潜在类别 CO 与推理时 novel 类别 Cu 存在语义重叠
2. 在计算 novel 类别概率时，分母中的 Σo (背景类别贡献) 会导致 novel 概率被低估
3. 用 alpha(o|x) = softmax(cos(w(x), t_o)) 表示 proposal 属于每个背景类别的概率
4. 用 beta(c|o) = softmax(t_o @ t_c.T) 表示背景类别 o 与 novel 类别 c 的语义相似度
5. 修正后: Σo_tilde = Σo - alpha(o|x) * beta(c|o) * exp(cos(w(x), t_c))
   (即从分母中减去与 novel 类别重叠的部分)
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ovdet.methods.builder import OVD


@OVD.register_module()
class IPR(nn.Module):
    def __init__(self, rectification_factor=0.3, num_bg_clusters=6, **kwargs):
        super().__init__()
        self.rectification_factor = rectification_factor
        self.num_bg_clusters = num_bg_clusters
        self.bcp_module = None

    def get_bg_cluster_centers(self):
        """Get cluster centers from BCP module (contextual embeddings of CO)."""
        if self.bcp_module is not None:
            cc = getattr(self.bcp_module, 'cluster_centers', None)
            if cc is not None and cc.shape[0] > 0:
                return cc
        return None

    def get_bg_prompts(self):
        """Get learnable background prompts from BCP module."""
        if self.bcp_module is not None:
            bg_prompts = getattr(self.bcp_module, 'bg_prompts', None)
            if bg_prompts is not None:
                return bg_prompts
        return None

    def rectify(self, cls_score, region_features, base_novel_cls_embeddings,
                bg_cluster_centers=None, bg_prompts_encoded=None):
        """Rectify probability scores per paper Eq.(18).

        Key idea:
        - alpha(o|x) = softmax(cos(w(x), t_o))  -- proposal x over bg cluster o
        - beta(c|o) = softmax(t_o @ t_c.T)       -- bg cluster o vs class c
        - redistribution = alpha @ beta * bg_score * factor
        """
        if bg_cluster_centers is None:
            bg_cluster_centers = self.get_bg_cluster_centers()
        if bg_cluster_centers is None or bg_cluster_centers.shape[0] == 0:
            return cls_score

        if not self.training and cls_score.shape[1] > base_novel_cls_embeddings.shape[0]:
            bg_score = cls_score[:, -1:]
            novel_base_scores = cls_score[:, :-1]

            if region_features.dim() == 3:
                region_embed = region_features.mean(dim=1)
            else:
                region_embed = region_features

            # alpha(o|x) = softmax(cos(w(x), t_o))
            region_to_cluster = region_embed @ bg_cluster_centers.T
            alpha = F.softmax(region_to_cluster, dim=-1)

            # beta(c|o) = softmax(t_o @ t_c.T)
            cluster_to_class = bg_cluster_centers @ base_novel_cls_embeddings.T
            beta = F.softmax(cluster_to_class, dim=-1)

            # redistribution = sum_o [alpha(o|x) * beta(c|o)] * bg_score * factor
            redistribution = alpha @ beta
            redistribution = redistribution * bg_score * self.rectification_factor

            novel_base_scores = novel_base_scores + redistribution
            bg_score = bg_score * (1.0 - self.rectification_factor)
            cls_score = torch.cat([novel_base_scores, bg_score], dim=-1)

        return cls_score