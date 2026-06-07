"""Utility functions for LBP modules (BCP, BOD, IPR)."""
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox2roi


class BackgroundFeatureQueue:
    """Bounded FIFO queue storing background proposal CLIP features."""
    def __init__(self, queue_size: int = 2048, feat_dim: int = 512):
        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.queue = torch.zeros(0, feat_dim)
        self._valid = False

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor):
        if features.shape[0] == 0:
            return
        features = features.detach().cpu()
        combined = torch.cat([self.queue, features], dim=0)
        if combined.shape[0] > self.queue_size:
            combined = combined[-self.queue_size:]
        self.queue = combined
        self._valid = True

    @torch.no_grad()
    def get_all(self) -> torch.Tensor:
        return self.queue

    def is_valid(self) -> bool:
        return self._valid and self.queue.shape[0] >= 10


class BGProposalSelector:
    """Select background proposals from RPN results."""
    def __init__(self, bg_objectness_thr: float = 0.5, min_proposals: int = 10):
        self.bg_objectness_thr = bg_objectness_thr
        self.min_proposals = min_proposals

    def filter_bg_proposals(self, proposals, gt_bboxes, ioa_thr=0.02):
        device = proposals.bboxes.device
        bg_mask = proposals.scores < self.bg_objectness_thr
        if bg_mask.sum() < self.min_proposals:
            k = min(self.min_proposals, len(proposals))
            _, indices = proposals.scores.topk(k, largest=False)
            bg_mask = torch.zeros(len(proposals), dtype=torch.bool, device=device)
            bg_mask[indices] = True
        bg_proposals = proposals[bg_mask]
        if len(gt_bboxes) > 0 and len(bg_proposals) > 0:
            from mmdet.structures.bbox import bbox_overlaps
            ioa = bbox_overlaps(gt_bboxes, bg_proposals.bboxes, mode='iof', is_aligned=False)
            max_ioa = ioa.max(dim=0).values
            no_gt_mask = max_ioa < ioa_thr
            if no_gt_mask.sum() > 0:
                bg_proposals = bg_proposals[no_gt_mask]
        return bg_proposals


def extract_clip_image_features_for_boxes(images, boxes, clip_model, clip_data_preprocessor):
    from mmcv.ops import roi_align
    image_encoder = clip_model.image_encoder
    clip_input_size = image_encoder.input_resolution
    clip_images = clip_data_preprocessor({'inputs': images})['inputs']
    input_to_clip = roi_align(clip_images, bbox2roi([boxes]), (clip_input_size, clip_input_size), 1.0, 2, 'avg', True)
    clip_features = image_encoder.encode_image(input_to_clip, normalize=True, return_tokens=False)
    return clip_features


def kmeans_clustering(features, k, max_iter=100, tol=1e-4, use_faiss=False):
    if use_faiss:
        try:
            import faiss
            n, d = features.shape
            features_np = features.cpu().numpy().astype('float32')
            kmeans = faiss.Kmeans(d, k, niter=max_iter, gpu=features.is_cuda)
            kmeans.train(features_np)
            _, labels = kmeans.index.search(features_np, 1)
            centers = torch.from_numpy(kmeans.centroids).to(features.device)
            labels = torch.from_numpy(labels.flatten()).to(features.device)
            return centers, labels
        except ImportError:
            pass
    try:
        from sklearn.cluster import KMeans
        n, d = features.shape
        features_np = features.cpu().numpy().astype('float32')
        km = KMeans(n_clusters=min(k, n), max_iter=max_iter, tol=tol, random_state=0, n_init='auto')
        labels_np = km.fit_predict(features_np)
        centers = torch.from_numpy(km.cluster_centers_).to(features.device)
        labels = torch.from_numpy(labels_np.astype('int64')).to(features.device)
        return centers, labels
    except ImportError:
        return _kmeans_pytorch(features, k, max_iter, tol)


def _kmeans_pytorch(features, k, max_iter=100, tol=1e-4):
    n, d = features.shape
    k = min(k, n)
    centers = features.new_zeros(k, d)
    centers[0] = features[torch.randint(0, n, (1,))]
    for i in range(1, k):
        dists = torch.cdist(features, centers[:i])
        min_dists, _ = dists.min(dim=-1)
        probs = min_dists / (min_dists.sum() + 1e-12)
        centers[i] = features[torch.multinomial(probs, 1)]
    for it in range(max_iter):
        dists = torch.cdist(features, centers)
        labels = dists.argmin(dim=-1)
        new_centers = features.new_zeros(k, d)
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                new_centers[i] = features[mask].mean(dim=0)
            else:
                new_centers[i] = centers[i]
        shift = (centers - new_centers).norm(dim=-1).max()
        centers = new_centers
        if shift < tol:
            break
    return centers, labels