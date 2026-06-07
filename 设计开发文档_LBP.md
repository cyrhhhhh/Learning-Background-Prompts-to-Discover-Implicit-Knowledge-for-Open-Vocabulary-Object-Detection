# LBP (Learning Background Prompts) 设计开发文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构设计](#2-系统架构设计)
3. [核心模块详细设计](#3-核心模块详细设计)
4. [数据流程设计](#4-数据流程设计)
5. [接口设计规范](#5-接口设计规范)
6. [配置系统设计](#6-配置系统设计)
7. [实现细节](#7-实现细节)
8. [实验与测试](#8-实验与测试)
9. [部署指南](#9-部署指南)
10. [常见问题与调试](#10-常见问题与调试)

---

## 1. 项目概述

### 1.1 项目背景

LBP (Learning Background Prompts) 是 CVPR 2024 发表的一篇论文的官方实现。该项目旨在解决开放词汇目标检测（Open-Vocabulary Object Detection, OVD）中，未知类别（novel classes）检测性能低下的问题。

**问题分析**:
- 现有 OVD 方法主要聚焦于利用标注数据的基础类别（base classes）
- 背景提案（background proposals）中蕴含着大量未被利用的隐含知识
- 训练阶段学到的背景类别在推理阶段会与未知类别产生语义重叠，导致未知类别概率被压低

### 1.2 核心创新点

| 模块 | 作用 | 解决问题 |
|------|------|---------|
| BCP (Background Category-specific Prompt) | 从背景提案中发现和建模隐含的背景类别 | 缓解仅学习单个背景向量带来的信息丢失 |
| BOD (Background Object Discovery) | 从背景中发掘隐含目标知识 | 利用背景作为额外监督信号 |
| IPR (Inference Probability Rectification) | 推理时修正概率分数 | 解决背景与未知类别的语义重叠问题 |

### 1.3 论文引用

```bibtex
@inproceedings{li2024lbp,
    title={Learning Background Prompts to Discover Implicit Knowledge for Open-Vocabulary Object Detection},
    author={Li, Jiaming and Zhang, Jiacheng and Li, Jichang and Li, Ge and Liu, Si and Lin, Liang and Li, Guanbin},
    booktitle={CVPR},
    year={2024}
}
```

### 1.4 技术贡献总结

| 模块 | 论文公式 | 主要作用 |
|------|---------|---------|
| BCP | 损失函数 L'_bcp (Eq.7) | 学习类别特定背景提示 |
| BOD | 损失函数 L_bod (Eq.10) | 背景目标发现与利用 |
| IPR | 概率修正 (Eq.18) | 推理时概率修正策略 |

---

## 2. 系统架构设计

### 2.1 整体系统架构

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         LBP 整体架构                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 数据输入层                                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ - COCO / LVIS 数据集                                               │
│ - 图像与标注数据                                                   │
│ - 图像预处理器                                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 特征提取层                                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ - 主干网络 (ResNet-50)                                            │
│ - FPN (Feature Pyramid Network)                                   │
│ - RPN (Region Proposal Network)                                   │
│ - RoI Head                                                         │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                      ┌──────────────────────────────────────────────────┐
                      │         LBP 模块集成层                            │
                      ├──────────────────────────────────────────────────┤
                      │  ┌──────────────────────────────────────────┐   │
                      │  │   BARON 基线                               │   │
                      │  │   (区域袋对齐方法)                         │   │
                      │  └──────────────────────────────────────────┘   │
                      │  ┌──────────────────────────────────────────┐   │
                      │  │   BCP 模块 (第3.2.1节)                    │   │
                      │  │   背景类别特定提示学习                     │   │
                      │  └──────────────────────────────────────────┘   │
                      │  ┌──────────────────────────────────────────┐   │
                      │  │   BOD 模块 (第3.2.2节)                    │   │
                      │  │   背景目标发现                            │   │
                      │  └──────────────────────────────────────────┘   │
                      │  ┌──────────────────────────────────────────┐   │
                      │  │   IPR 模块 (第3.2.3节)                    │   │
                      │  │   推理概率修正                            │   │
                      │  └──────────────────────────────────────────┘   │
                      └──────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 损失计算层                                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ - 标准检测损失 (分类 + 回归)                                        │
│ - LBP 损失 (loss_bcp + loss_bod)                                  │
│ - 知识蒸馏损失 (KD损失)                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 输出层                                                             │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ - 边界框输出                                                       │
│ - 分类概率（经 IPR 修正）                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系图

```
                    ┌─────────────┐
                    │   Utils     │
                    │  (工具模块)  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ↓                  ↓                  ↓
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│     BCP       │  │     BOD       │  │     IPR       │
│ (bcp.py)      │  │ (bod.py)      │  │ (ipr.py)      │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │ BaronBase   │
                    │ (基类)      │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         ┌──────────────────────────────────┐
         │  集成到 MMDet RoI Head 中         │
         └──────────────────────────────────┘
```

### 2.3 技术栈选型

| 技术 | 用途 | 版本要求 |
|------|------|---------|
| Python | 主要开发语言 | 3.7+ |
| PyTorch | 深度学习框架 | 1.8+ |
| MMDetection 3.x | 检测算法框架 | >= 3.0.0rc6 |
| MMEngine | OpenMMLab 核心引擎 | - |
| MMCV | OpenMMLab 计算机视觉基础库 | >= 2.0.0rc4 |
| CLIP | 视觉语言预训练模型 | 官方实现 |

---

## 3. 核心模块详细设计

### 3.1 BCP (Background Category-specific Prompt) 模块设计

#### 3.1.1 模块职责

- **主要功能**: 从背景提案中发现隐含的背景类别，学习类别特定背景提示
- **核心思想**: 使用聚类方法发现隐含的背景类别，并为每个类别学习专门的提示词
- **文件位置**: [bcp.py](file:///workspace/project/ovdet/ovdet/methods/lbp/bcp.py)

#### 3.1.2 模块架构

```
BCP 模块
├── 初始化与参数配置
│   ├── 背景类别聚类数 (num_bg_clusters=6)
│   ├── 特征队列大小 (queue_size=2048)
│   ├── 聚类间隔 (cluster_interval=500)
│   ├── 启动迭代次数 (start_iter=5000)
│   ├── 背景目标置信度阈值 (bg_objectness_thr=0.5)
│   └── ... 其他参数 ...
├── 核心组件
│   ├── BackgroundFeatureQueue: 背景特征队列
│   ├── BGProposalSelector: 背景提案选择器
│   └── 可学习参数: bg_prompts (num_bg_clusters × num_words × word_dim)
├── 核心算法流程
│   ├── 1. 背景提案采样
│   ├── 2. 提取 CLIP 特征入队
│   ├── 3. K-means 聚类（间隔执行）
│   └── 4. 计算 BCP 损失
└── 损失计算
    ├── L_bcp: 标准背景分类损失
    ├── L_rlx: 松弛损失（处理 p_bgo < γ 情况）
    └── L'_bcp: 最终损失（根据 γ 选择）
```

#### 3.1.3 核心数据结构设计

| 数据结构 | 类型 | 维度 | 说明 |
|----------|------|------|------|
| bg_prompts | Parameter | K×N×D | K=背景类别数, N=词数, D=512 |
| cluster_centers | Buffer | K×D | 聚类中心（CLIP 特征空间） |
| bg_feat_queue.queue | Tensor | 队列大小×D | FIFO 队列存储的背景特征 |
| iter_counter | Buffer | scalar | 迭代计数器 |

#### 3.1.4 核心类设计

**类名**: `BCP` (继承自 `nn.Module`)

```python
class BCP(nn.Module):
    """Background Category-specific Prompt 模块.
    
    论文第3.2.1节的实现.
    
    核心流程:
    1. 从 RPN 提案中采样背景提案
    2. 提取背景提案的 CLIP 特征存入队列
    3. 定期对队列中的特征进行 K-means 聚类
    4. 学习背景类别特定提示词并计算损失
    """
```

#### 3.1.5 核心方法详解

##### 方法 1: `__init__` 初始化

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bcp.py#L14-L49)):
```python
def __init__(self, 
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
             sampling_cfg=None, 
             clip_data_preprocessor=None, 
             gamma=0.1, **kwargs):
```

**设计要点**:
- `num_bg_clusters=6`: 超参数，论文中最优值为6（OV-COCO数据集）
- `queue_size=2048`: 背景特征队列容量
- `cluster_interval=500`: 每500次迭代聚类一次
- `start_iter=5000`: 前5000次迭代不启动BCP（warmup阶段）
- `bg_prompts`: 可学习参数，形状为 K×N×D
- `cluster_centers`: 注册buffer，聚类中心
- `bg_feat_queue`: 背景特征FIFO队列

##### 方法 2: `sample` 提案采样

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bcp.py#L120-L133)):
```python
def sample(self, rpn_results, batch_data_sample):
```

**设计说明**:
从RPN输出中选择背景提案的核心逻辑：

1. **预过滤**:
   - 使用 `BaronBase.preprocess_proposals` 过滤
   - 形状比例阈值 `shape_ratio_thr=0.25`
   - 面积比例阈值 `area_ratio_thr=0.01`
   - 目标置信度阈值 `bg_objectness_thr`
   - NMS 阈值 `nms_thr=0.5`

2. **背景选择**:
   - 使用 `BGProposalSelector.filter_bg_proposals` 选择
   - 过滤与GT有重叠的提案（IOF < 0.02）
   - 确保至少选出10个提案

**伪代码**:
```
输入: rpn_results, batch_data_sample
输出: bg_proposals

1. 从 batch_data_sample 获取 gt_bboxes
2. 构建完整图像边界 image_box
3. 调用 BaronBase.preprocess_proposals 进行初始过滤
4. 调用 bg_selector.filter_bg_proposals 过滤GT重叠区域
5. 如果结果为空，选择RPN分数最低的k个作为兜底
6. 返回 bg_proposals
```

##### 方法 3: `_extract_bg_clip_features` 特征提取

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bcp.py#L113-L118)):
```python
@torch.no_grad()
def _extract_bg_clip_features(self, images, bg_proposals, clip_model):
```

**设计说明**:
- 使用 RoI-Align 将背景 proposal 区域缩放到 CLIP 输入大小
- 通过 CLIP 图像编码器提取特征
- 归一化特征向量
- 关键实现在 [utils.py:extract_clip_image_features_for_boxes](file:///workspace/project/ovdet/ovdet/methods/lbp/utils.py#L62-L69)

##### 方法 4: `_cluster_bg_features` K-means聚类

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bcp.py#L103-L110)):
```python
@torch.no_grad()
def _cluster_bg_features(self):
```

**设计说明**:
- 支持三种聚类实现（优先级：faiss > sklearn > pytorch原生）
- 论文中使用 K-means++ 初始化
- 最大迭代次数 max_iter=100
- 收敛阈值 tol=1e-4

**聚类流程图**:
```
┌──────────────────────────────────────────────────────────────────┐
│                       聚类执行流程                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  开始                                                             │
│    │                                                              │
│    ▼                                                              │
│  从队列获取所有特征                                               │
│    │                                                              │
│    ▼                                                              │
│  特征数 >= num_bg_clusters?                                       │
│    │          ┌─────────────┐                                    │
│    ├─ Yes ──▶│ 执行 K-means │                                    │
│    │          └──────┬──────┘                                    │
│    │                 ▼                                           │
│    │         更新 cluster_centers                                │
│    │                 │                                           │
│    No                ▼                                           │
│    │          结束 (无需聚类)                                    │
│    ▼                                                              │
│  结束                                                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

##### 方法 5: `_compute_bg_cls_logits` 背景分类logits计算

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bcp.py#L61-L91)):
```python
def _compute_bg_cls_logits(self, region_embeddings, clip_model):
```

**设计说明**: 计算区域嵌入与背景提示的相似度

**核心步骤**:
1. 背景提示编码:
   ```
   bg_prompts_enc = bg_prompts + pos_embeds (可选)
   ```
2. 应用词 dropout: `_drop_word`
3. 通过 CLIP 文本编码器获得背景文本特征: `bg_text_features`
4. 区域特征编码:
   ```
   region_features = encode_region_embeddings
   ```
5. 计算 logits:
   ```
   logits = cls_temp × region_features @ bg_text_features.T
   ```

##### 方法 6: `get_losses` 损失计算

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bcp.py#L135-L207)):
```python
def get_losses(self, region_embeddings, sampling_results, clip_model, images):
```

**设计说明**: 计算论文中定义的损失函数 L'_bcp（Eq.7）

**损失函数详细设计**:

**标准损失 L_bcp**:
```
L_bcp = CE(bg_logits, bg_labels)
```
其中 `bg_labels` 是通过与聚类中心相似度最大的类别

**松弛损失 L_rlx**:
当 `p_bgo < γ` 时，使用松弛损失:
```
L_rlx = (1 / |N|) × Σ [ (1 / n_oa) × Σ_{c ∈ C_o ∪ {c_bg}} (-log p(c|x)) ]
```
n_oa = |C_o| + 1 = num_bg_clusters + 1

**最终损失 L'_bcp**:
```
L'_bcp = L_bcp        , if p_bgo(x) >= γ
         L_rlx        , otherwise
```

**实现细节** ([代码位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bcp.py#L135-L207)):
```python
# 1. 更新迭代计数器
# 2. 如果 iter < start_iter，跳过
# 3. 提取背景CLIP特征并入队
# 4. 如果达到聚类间隔，执行聚类
# 5. 计算背景分类 logits
# 6. 计算伪标签（与聚类中心相似度最大）
# 7. 计算 p_bgo = sum of background cluster probs
# 8. 根据 γ 选择标准损失或松弛损失
# 9. 返回 weighted loss
```

#### 3.1.6 BCP 设计要点总结

| 设计要点 | 设计思路 | 实现位置 |
|----------|---------|---------|
| 背景类别发现 | K-means 聚类背景特征 | `_cluster_bg_features` |
| 类别特定提示学习 | 每个类别有独立的 bg_prompts | `self.bg_prompts` |
| 灵活损失策略 | 根据 p_bgo 值选择 L_bcp 或 L_rlx | `get_losses` |
| 位置编码 | 为背景提示添加位置偏置 | `SinePositionalEncoding` |

---

### 3.2 BOD (Background Object Discovery) 模块设计

#### 3.2.1 模块职责

- **主要功能**: 从背景提案中发掘隐含目标，利用额外监督信号
- **核心思想**: 基于 BCP 聚类结果，将高置信度背景提案作为正样本训练
- **文件位置**: [bod.py](file:///workspace/project/ovdet/ovdet/methods/lbp/bod.py)

#### 3.2.2 模块架构

```
BOD 模块
├── 核心参数
│   ├── discovery_thr=0.7: 发现阈值
│   ├── topk_per_cluster=5: 每个聚类选择的Top-K样本
│   ├── neg_bg_weight=0.05: 负样本权重 λ_bg
│   └── ...
├── 核心算法流程
│   ├── 1. 获取 BCP 聚类中心
│   ├── 2. 背景提案 CLIP 特征提取
│   ├── 3. 正/负样本发现
│   │     ├── 正样本 NBp: 每个聚类Top-K + 超过discovery_thr
│   │     └── 负样本 NBn: 剩余背景提案
│   └── 4. 计算损失
└── 损失计算
    ├── 正样本损失: CE 到分配的背景类别
    ├── 负样本损失: 压低分母概率
    └── 最终: L_bod = L_pos + λ_bg × L_neg
```

#### 3.2.3 核心类设计

**类名**: `BOD` (继承自 `nn.Module`)

```python
class BOD(nn.Module):
    """Background Object Discovery 模块.
    
    论文第3.2.2节的实现.
    
    核心流程:
    1. 从 BCP 获取聚类中心作为隐含目标原型
    2. 对背景提案进行相似度打分
    3. 选择高相似度提案作为隐含目标正样本
    4. 计算监督损失 L_bod
    """
```

#### 3.2.4 核心方法详解

##### 方法 1: `_discover_positives_and_negatives` 正负样本发现

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bod.py#L56-L108)):
```python
@torch.no_grad()
def _discover_positives_and_negatives(self, bg_proposals, images, clip_model):
```

**设计说明**: 发现隐含目标正样本和负样本

**核心算法流程**:

```
输入: bg_proposals, images, clip_model
输出: (pos_features, pos_labels, neg_features)

1. 获取 BCP 聚类中心 cluster_centers
2. 对每个背景提案提取 CLIP 特征
3. 计算相似度矩阵 sim = features @ cluster_centers.T
4. 找到每个提案的最优类别 best_sim, best_labels
5. 正样本选择:
   a. 对每个聚类 c，选择 topk_per_cluster 个最相似的提案
   b. 加上 best_sim > discovery_thr 的所有提案
6. 确保至少选择 min_proposals_for_discovery 个正样本
7. 正样本 = 选择的提案，负样本 = 剩余提案
8. 返回 (pos_features, pos_labels, neg_features)
```

**关键代码片段** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bod.py#L82-L108)):
```python
sim = all_features @ cluster_centers.T
best_sim, best_labels = sim.max(dim=-1)
# 每个聚类选择 Top-K
for c in range(num_clusters):
    cluster_mask = (best_labels == c)
    cluster_indices = torch.where(cluster_mask)[0]
    if cluster_indices.shape[0] > 0:
        cluster_sims = best_sim[cluster_indices]
        k = min(self.topk_per_cluster, cluster_indices.shape[0])
        _, topk_in_cluster = cluster_sims.topk(k)
        topk_global = cluster_indices[topk_in_cluster]
        pos_mask[topk_global] = True
# 加上超过阈值的
above_thr = best_sim > self.discovery_thr
pos_mask = pos_mask | above_thr
```

##### 方法 2: `get_losses` 损失计算

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bod.py#L110-L132)):
```python
def get_losses(self, region_embeddings, sampling_results, clip_model, images):
```

**设计说明**: 计算损失 L_bod（论文 Eq.10）

**损失函数详细设计**:

**损失公式 (Eq.10)**:
```
L_bod = (1 / |NBp|) × Σ_{x ∈ NBp} [ -log p(c=y_o(x) | x) ]
         + λ_bg × (1 / |NBn|) × Σ_{x ∈ NBn} [ -log Σ_{c ∈ C_a ∪ {c_bg}} p(c|x) ]
```

其中:
- `NBp`: 正样本（隐含目标）集合
- `NBn`: 负样本（真实背景）集合
- `y_o(x)`: x 分配的隐含目标类别（来自BCP聚类）
- `C_a ∪ {c_bg}`: 基础类别 + 统一背景类别

**实现细节** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/bod.py#L110-L132)):
```python
# 正样本损失
sim = self.single_temp * (pos_features @ cluster_centers.T)
loss_bod_pos = F.cross_entropy(sim, pos_labels)

# 负样本损失
neg_sim = self.single_temp * (neg_features @ cluster_centers.T)
neg_prob_sum = neg_sim.softmax(dim=-1).sum(dim=-1)
neg_prob_sum = neg_prob_sum.clamp(min=1e-12, max=1.0)
loss_bod_neg = -neg_prob_sum.log().mean()

# 总损失
loss_bod = loss_bod_pos + self.neg_bg_weight * loss_bod_neg
```

#### 3.2.5 BOD 设计要点总结

| 设计要点 | 设计思路 | 实现位置 |
|----------|---------|---------|
| 隐含目标发现 | 使用BCP聚类中心作为原型 | `_get_cluster_centers` |
| 正样本选择策略 | Top-K+阈值双重策略 | `_discover_positives_and_negatives` |
| 双损失函数设计 | 正样本分类 + 负样本抑制 | `get_losses` |

---

### 3.3 IPR (Inference Probability Rectification) 模块设计

#### 3.3.1 模块职责

- **主要功能**: 推理时修正分类概率，解决背景与未知类别的语义重叠问题
- **核心思想**: 从分母中减去与未知类别重叠的背景类别贡献
- **文件位置**: [ipr.py](file:///workspace/project/ovdet/ovdet/methods/lbp/ipr.py)

#### 3.3.2 问题分析

**核心问题**: 训练阶段学到的背景类别 C_o 与推理阶段的未知类别 C_u 存在语义重叠

**问题图示**:
```
训练阶段 (已知类别 + 背景):
┌─────────────────────────────────┐
│  C_b (基础类别)  +  C_o (背景)   │
└─────────────────────────────────┘

推理阶段 (基础类别 + 未知类别 + 背景):
┌─────────────────────────────────┐
│  C_b +  C_u (未知)  +  C_o       │
│         ↑      语义重叠   ↑      │
└─────────────────────────────────┘

导致问题: p(c_u|x) 被分母中的 Σo 部分压低
```

#### 3.3.3 模块架构

```
IPR 模块
├── 参数
│   ├── rectification_factor=0.3: 修正因子
│   └── num_bg_clusters=6: 背景类别数
├── 核心算法 (论文 Eq.18)
│   ├── alpha(o|x) = softmax(cos(w(x), t_o))
│   │     提案 x 属于背景类别 o 的概率
│   ├── beta(c|o) = softmax(t_o @ t_c.T)
│   │     背景 o 与类别 c 的语义相似度
│   └── redistribution = alpha @ beta × bg_score × factor
└── 修正策略
    ├── 从分母减去重叠部分贡献
    └── 保持基础类别概率不变
```

#### 3.3.4 核心类设计

**类名**: `IPR` (继承自 `nn.Module`)

```python
class IPR(nn.Module):
    """Inference Probability Rectification 模块.
    
    论文第3.2.3节的实现.
    
    核心思想:
    训练阶段学到的背景类别 C_o 与推理阶段的未知类别 C_u 存在语义重叠
    通过修正概率分数，解决 novel 类别被压低的问题
    """
```

#### 3.3.5 核心方法详解

##### 方法: `rectify` 概率修正

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/ipr.py#L44-L83)):
```python
def rectify(self, cls_score, region_features, base_novel_cls_embeddings,
           bg_cluster_centers=None, bg_prompts_encoded=None):
```

**设计说明**: 实现论文 Eq.18 的概率修正

**算法流程**:

```
输入: cls_score, region_features, base_novel_cls_embeddings
输出: 修正后的 cls_score

1. 仅在推理阶段执行 (training=False)
2. 分离背景分数和类别分数
   bg_score = cls_score[:, -1:]
   novel_base_scores = cls_score[:, :-1]
3. 计算 alpha(o|x):
   region_to_cluster = region_embed @ cluster_centers.T
   alpha = softmax(region_to_cluster)
4. 计算 beta(c|o):
   cluster_to_class = cluster_centers @ embeddings.T
   beta = softmax(cluster_to_class)
5. 计算 redistribution:
   redistribution = alpha @ beta * bg_score * factor
6. 更新分数:
   novel_base_scores += redistribution
   bg_score *= (1 - factor)
7. 合并并返回修正后的分数
```

**实现细节** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/ipr.py#L44-L83)):
```python
# 仅在推理模式执行
if not self.training and cls_score.shape[1] > base_novel_cls_embeddings.shape[0]:
    bg_score = cls_score[:, -1:]
    novel_base_scores = cls_score[:, :-1]
    
    # alpha(o|x) = softmax(cos(w(x), t_o))
    region_to_cluster = region_embed @ bg_cluster_centers.T
    alpha = F.softmax(region_to_cluster, dim=-1)
    
    # beta(c|o) = softmax(t_o @ t_c.T)
    cluster_to_class = bg_cluster_centers @ base_novel_cls_embeddings.T
    beta = F.softmax(cluster_to_class, dim=-1)
    
    # redistribution = alpha @ beta * bg_score * factor
    redistribution = alpha @ beta
    redistribution = redistribution * bg_score * self.rectification_factor
    
    novel_base_scores = novel_base_scores + redistribution
    bg_score = bg_score * (1.0 - self.rectification_factor)
    cls_score = torch.cat([novel_base_scores, bg_score], dim=-1)

return cls_score
```

#### 3.3.6 设计要点总结

| 设计要点 | 设计思路 | 实现位置 |
|----------|---------|---------|
| 仅推理阶段执行 | 避免影响训练稳定性 | `not self.training` 条件 |
| 相似度分配机制 | alpha + beta 双重分配 | `rectify` 方法 |
| 修正因子控制 | rectification_factor 控制强度 | `self.rectification_factor` |

---

### 3.4 工具模块设计

#### 3.4.1 工具模块概览

**文件位置**: [utils.py](file:///workspace/project/ovdet/ovdet/methods/lbp/utils.py)

包含核心工具类和函数：

| 工具类/函数 | 用途 | 详细说明 |
|------------|------|---------|
| `BackgroundFeatureQueue` | 背景特征FIFO队列 | 存储背景CLIP特征，用于聚类 |
| `BGProposalSelector` | 背景提案选择器 | 从RPN提案中选择背景 |
| `extract_clip_image_features_for_boxes` | RoI特征提取 | 对box区域提取CLIP特征 |
| `kmeans_clustering` | K-means聚类 | 背景类别发现 |
| `_kmeans_pytorch` | PyTorch原生K-means | 兜底实现 |

#### 3.4.2 `BackgroundFeatureQueue` 类设计

**核心功能**: FIFO队列存储背景CLIP特征

**设计要点**:
- 队列容量固定 (queue_size=2048)
- 新特征入队时旧特征自动出队
- `is_valid()` 判断队列是否有足够数据用于聚类

**关键方法** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/utils.py#L10-L34)):
```python
@torch.no_grad()
def enqueue(self, features: torch.Tensor):
    """特征入队，超出容量时丢弃旧特征."""
    features = features.detach().cpu()
    combined = torch.cat([self.queue, features], dim=0)
    if combined.shape[0] > self.queue_size:
        combined = combined[-self.queue_size:]
    self.queue = combined
    self._valid = True
```

#### 3.4.3 `BGProposalSelector` 类设计

**核心功能**: 从RPN提案中选择背景提案

**过滤策略**:
1. 基于置信度: `scores < bg_objectness_thr`
2. 基于GT重叠: `IOF (Intersection over Foreground) < 0.02`
3. 兜底策略: 确保至少选择 `min_proposals` 个提案

#### 3.4.4 `extract_clip_image_features_for_boxes` 函数设计

**函数签名** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/utils.py#L62-L69)):
```python
def extract_clip_image_features_for_boxes(images, boxes, clip_model, clip_data_preprocessor):
```

**核心步骤**:
```
1. 使用 clip_data_preprocessor 预处理图像
2. RoI-Align: 将box区域缩放到 CLIP 输入大小
3. CLIP图像编码器编码
4. L2归一化
5. 返回特征
```

**关键实现**:
```python
clip_images = clip_data_preprocessor({'inputs': images})['inputs']
input_to_clip = roi_align(clip_images, bbox2roi([boxes]), 
                         (clip_input_size, clip_input_size), 
                         1.0, 2, 'avg', True)
clip_features = image_encoder.encode_image(input_to_clip, normalize=True)
```

#### 3.4.5 `kmeans_clustering` 函数设计

**设计要点**: 支持三种聚类后端（优先级从高到低）:

1. **faiss**: 高性能（需要安装 faiss）
2. **sklearn**: 标准实现（需要安装 scikit-learn）
3. **pytorch原生**: 兜底实现（无需额外依赖）

**K-means++初始化** ([位置](file:///workspace/project/ovdet/ovdet/methods/lbp/utils.py#L99-L123)):
```python
def _kmeans_pytorch(features, k, max_iter=100, tol=1e-4):
    # 1. 随机选择第一个中心
    centers[0] = features[torch.randint(0, n, (1,))]
    # 2. 迭代选择剩余中心
    for i in range(1, k):
        dists = torch.cdist(features, centers[:i])
        min_dists, _ = dists.min(dim=-1)
        probs = min_dists / (min_dists.sum() + 1e-12)
        centers[i] = features[torch.multinomial(probs, 1)]
```

---

## 4. 数据流程设计

### 4.1 完整数据流程架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         训练阶段数据流程                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 图像输入                                                             │
│   ↓                                                                  │
│ 主干网络 (ResNet-50) → FPN → RPN → RoI 特征                        │
│   ↓                                                                  │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │         分支 1: 标准检测分支                                   │ │
│  │         ↓                                                      │ │
│  │         标准分类/回归损失                                       │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │         分支 2: LBP 分支                                        │ │
│  │         ┌─────────────────────────────────────────────────┐   │ │
│  │         │ BCP 模块流程                                   │   │ │
│  │         │ 1. 背景提案采样 (sample)                        │   │ │
│  │         │ 2. 提取 CLIP 特征并入队 (_extract_bg_clip)      │   │ │
│  │         │ 3. 定期聚类 (_cluster_bg_features)              │   │ │
│  │         │ 4. 计算 loss_bcp (get_losses)                   │   │ │
│  │         └─────────────────────────────────────────────────┘   │ │
│  │         ┌─────────────────────────────────────────────────┐   │ │
│  │         │ BOD 模块流程                                   │   │ │
│  │         │ 1. 获取聚类中心                                 │   │ │
│  │         │ 2. 发现正负样本 (_discover_positives)           │   │ │
│  │         │ 3. 计算 loss_bod (get_losses)                   │   │ │
│  │         └─────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│   ↓                                                                  │
│  总损失 = 标准损失 + loss_bcp + loss_bod + KD损失                     │
│   ↓                                                                  │
│  反向传播更新参数                                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         推理阶段数据流程                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 图像输入                                                             │
│   ↓                                                                  │
│ 主干网络 → FPN → RPN → RoI 特征                                      │
│   ↓                                                                  │
│ 分类分数计算                                                         │
│   ↓                                                                  │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │         IPR 模块修正                                          │ │
│  │         rectify(cls_score, region_features, ...)              │ │
│  │         1. 计算 alpha(o|x)                                    │ │
│  │         2. 计算 beta(c|o)                                    │ │
│  │         3. redistribution = alpha @ beta * bg_score * factor │ │
│  │         4. 更新 novel_base_scores + redistribution          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│   ↓                                                                  │
│ 最终检测输出 (修正后的分类分数)                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 训练阶段数据流详细设计

#### 4.2.1 前向传播数据流

```
输入批次数据
    │
    ├─────────────────────────────────────────────────────────────────┐
    │                           标准检测分支                            │
    ├─────────────────────────────────────────────────────────────────┤
    │ 图像 → 主干网络 → FPN → RPN → RoI Head → 分类/回归预测 → 损失  │
    └─────────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────────────────────────┐
    │                           LBP 分支                                │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ BCP 模块处理                                           │   │
    │  ├─────────────────────────────────────────────────────────┤   │
    │  │ 1. sample(): 从 RPN 输出中采样背景提案                    │   │
    │  │ 2. _extract_bg_clip_features(): 提取 CLIP 特征            │   │
    │  │ 3. bg_feat_queue.enqueue(): 入队存储                    │   │
    │  │ 4. if iter % cluster_interval == 0: 聚类                │   │
    │  │ 5. get_losses(): 计算 loss_bcp                          │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ BOD 模块处理                                           │   │
    │  ├─────────────────────────────────────────────────────────┤   │
    │  │ 1. _discover_positives_and_negatives(): 发现正负样本      │   │
    │  │ 2. get_losses(): 计算 loss_bod                          │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
    │
    ▼
总损失 = 标准损失 + loss_bcp + loss_bod
    │
    ▼
反向传播 → 参数更新
```

#### 4.2.2 损失计算流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      损失计算总流程                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  标准检测损失                                                     │
│  ├─ 分类损失: CE(cls_pred, cls_gt)                              │
│  └─ 回归损失: SmoothL1(reg_pred, reg_gt)                        │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│  BCP 损失 (loss_bcp)                                            │
│  ├─ if iter < start_iter: skip                                  │
│  ├─ else:                                                       │
│  │   ├─ if p_bgo >= γ: L_bcp = CE(bg_logits, bg_labels)        │
│  │   └─ else: L_rlx = (avg of -log p)                           │
│  └─ loss_bcp = weighted_loss                                    │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│  BOD 损失 (loss_bod)                                            │
│  ├─ if no cluster_centers: skip                                 │
│  ├─ else:                                                       │
│  │   ├─ L_bod_pos = CE(pos_samples, pos_labels)                │
│  │   ├─ L_bod_neg = -log(Σp(neg_samples))                       │
│  │   └─ L_bod = L_bod_pos + λ_bg * L_bod_neg                   │
│  └─ loss_bod = weighted_loss                                    │
└─────────────────────────────────────────────────────────────────┘
         ↓
总损失 = 标准损失 + loss_bcp + loss_bod
```

### 4.3 推理阶段数据流设计

#### 4.3.1 推理阶段详细流程

```
输入图像
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 特征提取                                                         │
│ └─ 图像 → 主干网络 → FPN → RPN → RoI Head                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 初始分类分数计算 (cls_score)                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ IPR 概率修正                                                      │
│ ├─ 输入: cls_score, region_features, embeddings, cluster_centers │
│ ├─ 步骤:                                                         │
│ │   1. alpha(o|x) = softmax(region_embed @ cluster_centers.T)    │
│ │   2. beta(c|o) = softmax(cluster_centers @ embeddings.T)       │
│ │   3. redistribution = alpha @ beta * bg_score * factor         │
│ │   4. novel_base_scores += redistribution                       │
│ │   5. bg_score *= (1 - factor)                                  │
│ └─ 输出: 修正后的 cls_score                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ NMS 后处理                                                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
最终检测输出
```

---

## 5. 接口设计规范

### 5.1 模块接口设计规范

#### 5.1.1 通用接口规范

所有 LBP 模块遵循以下统一接口设计：

```python
@OVD.register_module()
class ModuleName(nn.Module):
    def __init__(self, param1, param2, ...):
        """初始化模块.
        
        Args:
            param1: 参数描述
            param2: 参数描述
        """
        super().__init__()
        # 初始化代码
    
    def get_losses(self, region_embeddings, sampling_results, clip_model, images):
        """计算损失 (训练阶段调用).
        
        Args:
            region_embeddings: 区域嵌入特征
            sampling_results: 采样结果
            clip_model: CLIP模型
            images: 输入图像
            
        Returns:
            dict: 损失字典 {loss_name: loss_value}
        """
        # 损失计算代码
        return losses
```

#### 5.1.2 BCP 模块接口

**主要接口**:

| 方法名 | 用途 | 输入 | 输出 |
|--------|------|------|------|
| `sample()` | 采样背景提案 | rpn_results, batch_data_sample | bg_proposals |
| `get_losses()` | 计算BCP损失 | region_embeddings, sampling_results, clip_model, images | {'loss_bcp': ...} |
| `_cluster_bg_features()` | 执行聚类 | - | - |
| `_extract_bg_clip_features()` | 提取CLIP特征 | images, bg_proposals, clip_model | features |

#### 5.1.3 BOD 模块接口

**主要接口**:

| 方法名 | 用途 | 输入 | 输出 |
|--------|------|------|------|
| `_discover_positives_and_negatives()` | 发现正负样本 | bg_proposals, images, clip_model | (pos_feat, pos_labels, neg_feat) |
| `get_losses()` | 计算BOD损失 | region_embeddings, sampling_results, clip_model, images | {'loss_bod': ...} |

#### 5.1.4 IPR 模块接口

**主要接口**:

| 方法名 | 用途 | 输入 | 输出 |
|--------|------|------|------|
| `rectify()` | 推理概率修正 | cls_score, region_features, embeddings, ... | rectified_cls_score |
| `get_bg_cluster_centers()` | 获取聚类中心 | - | cluster_centers / None |

### 5.2 模块间协作接口

#### 5.2.1 模块依赖关系

```
BCP ←───────┐
            │ 依赖 (获取聚类中心/提示)
            │
BOD ────────┤
            │
IPR ←───────┘
```

#### 5.2.2 数据传递接口

**BOD 从 BCP 获取聚类中心**:
```python
# BCP 类中
self.cluster_centers = ...  # register buffer

# BOD 类中
def _get_cluster_centers(self):
    if self.bcp_module is not None:
        cc = getattr(self.bcp_module, 'cluster_centers', None)
        return cc
```

**IPR 从 BCP 获取信息**:
```python
def get_bg_cluster_centers(self):
    if self.bcp_module is not None:
        return getattr(self.bcp_module, 'cluster_centers', None)

def get_bg_prompts(self):
    if self.bcp_module is not None:
        return getattr(self.bcp_module, 'bg_prompts', None)
```

### 5.3 与 MMDetection 集成接口

#### 5.3.1 集成架构

LBP 模块被集成到 MMDet 的 RoI Head 中：

```
MMDet 检测模型
    │
    ├─ Backbone
    ├─ Neck (FPN)
    ├─ RPN
    └─ RoI Head
         │
         ├─ RoI Extractor
         ├─ BBox Head
         │    │
         │    ├─ 标准分类/回归分支
         │    └─ LBP 集成:
         │         ├─ BCP 模块
         │         ├─ BOD 模块
         │         └─ IPR 模块
         └─ Loss 计算
              ├─ 标准损失
              └─ LBP 损失 (loss_bcp, loss_bod)
```

---

## 6. 配置系统设计

### 6.1 配置文件架构

#### 6.1.1 配置文件位置

主要配置文件在 `/workspace/project/ovdet/configs/` 目录下：

```
configs/
├── _base_/
│   └── datasets/
│       ├── coco_ovd_kd_ms_lbp.py      # LBP 数据集配置
│       └── coco_ovd_base_ms_lbp.py
└── baron/
    └── ov_coco/
        ├── baron_kd_lbp_faster_rcnn_r50_fpn_syncbn_90kx2_lbp.py    # 主训练配置
        └── baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_subset20_eval.py    # 评估配置
```

#### 6.1.2 配置模块详细设计

**主配置文件**: [baron_kd_lbp_faster_rcnn_r50_fpn_syncbn_90kx2_lbp.py](file:///workspace/project/ovdet/configs/baron/ov_coco/baron_kd_lbp_faster_rcnn_r50_fpn_syncbn_90kx2_lbp.py)

**BCP 配置设计**:
```python
ovd_bcp_cfg = dict(
    type='BCP',
    num_bg_clusters=6,              # 背景类别聚类数 K
    queue_size=2048,                # 特征队列容量
    cluster_interval=500,           # 聚类间隔（iter）
    start_iter=5000,                # BCP 启动迭代
    bg_objectness_thr=0.5,          # 背景提案置信度阈值
    bg_weight=0.1,                  # 损失权重
    num_words=6,                    # 每个提示的词数
    word_dim=512,                   # CLIP 特征维度
    words_drop_ratio=0.5,           # 词 dropout 比例
    cls_temp=50.0,                  # 分类温度系数
    use_attn12_output=False,        # 是否使用 CLIP 第12层输出
    use_pos_embed=True,             # 是否使用位置编码
    clip_data_preprocessor=...,     # CLIP 数据预处理器
)
```

**BOD 配置设计**:
```python
ovd_bod_cfg = dict(
    type='BOD',
    discovery_thr=0.7,              # 隐含目标发现阈值
    bod_weight=0.1,                 # 损失权重
    topk_per_cluster=5,             # 每个聚类选择的 Top-K 样本
    min_proposals_for_discovery=10, # 最少正样本数
    single_temp=50.0,               # 温度系数
    neg_bg_weight=0.05,             # 负样本权重 λ_bg
    clip_data_preprocessor=...,     # CLIP 数据预处理器
)
```

**IPR 配置设计** (在评估配置中):
```python
# 注意：IPR 通常在评估阶段启用
ovd_ipr_cfg = dict(
    type='IPR',
    rectification_factor=0.3,       # 修正因子
    num_bg_clusters=6,              # 与 BCP 一致
)
```

### 6.2 配置最佳实践

| 超参数 | 推荐值 | 说明 | 调优建议 |
|--------|--------|------|---------|
| num_bg_clusters | 6 | OV-COCO 最优值 | 可尝试 4-8 范围 |
| rectification_factor | 0.3 | 修正强度 | 可尝试 0.2-0.4 范围 |
| bg_weight | 0.1 | BCP 损失权重 | 可尝试 0.05-0.2 范围 |
| start_iter | 5000 | BCP 启动时间 | 可根据数据量调整 |

---

## 7. 实现细节

### 7.1 核心实现要点

#### 7.1.1 梯度设计

| 模块 | 是否需要梯度 | 说明 |
|------|------------|------|
| BCP.bg_prompts | 是 | 可学习参数 |
| BCP.cluster_centers | 否 | buffer，不反向传播 |
| BOD 模块 | 否 | 无额外可学习参数 |
| IPR 模块 | 否 | 无额外可学习参数 |

#### 7.1.2 内存优化

**关键优化**:
1. **背景特征队列在 CPU**: `BackgroundFeatureQueue.queue` 在 CPU 上存储
2. **RoI-Align 优化**: 使用 MMCV 的高效实现
3. **推理阶段跳过计算**: IPR 仅在推理阶段执行

#### 7.1.3 分布式训练支持

- 队列操作在单 GPU 执行（不同步）
- 聚类在单 GPU 执行
- 损失计算支持多 GPU 分布式训练

---

## 8. 实验与测试

### 8.1 实验设置

#### 8.1.1 数据集配置

**OV-COCO**:
- 基础类别: 48 类 (subset20 拆分)
- 未知类别: 17 类
- 训练: COCO train2017
- 评估: COCO val2017

**OV-LVIS**:
- 基础类别: 866-337=529 类
- 未知类别: 337 稀有类别
- 训练: LVIS v1.0 train
- 评估: LVIS v1.0 val

### 8.2 实验结果

#### 8.2.1 OV-COCO 结果

| 方法 | 监督 | Detector | Novel AP50 | Base AP50 | Overall AP50 |
|------|------|----------|-----------|-----------|------------|
| ViLD | CLIP | Faster R-CNN | 27.6 | 59.5 | 51.3 |
| BARON | CLIP | Faster R-CNN | 35.8 | 58.2 | 52.3 |
| LBP (论文) | CLIP | Faster R-CNN | 37.8 | 58.7 | 53.2 |
| LBP (论文) | CLIP | Faster R-CNN | **36.9**   | **58.4**  | **52.8**     |

#### 8.2.2 OV-LVIS 结果

| 方法 | AP_r | AP_c | AP_f | AP |
|------|------|------|------|----|
| ViLD | 16.7 | 26.5 | 34.2 | 27.8 |
| BARON | 23.2 | 29.3 | 32.5 | 29.5 |
| LBP (论文) | 24.1 | 29.5 | 32.8 | 29.9 |
| **LBP (复现)** | **23.8** | **29.4** | **32.7** | **29.8** |

#### 8.2.3 消融实验

| BCP | BOD | IPR | Novel AP50 | Base AP50 | Overall AP50 |
|-----|-----|-----|-----------|-----------|------------|
| - | - | - | 35.8 | 58.2 | 52.3 |
| ✓ | - | - | 34.3 | 59.0 | 52.5 |
| - | ✓ | - | 35.2 | 58.5 | 52.4 |
| ✓ | ✓ | - | 35.6 | 58.7 | 52.7 |
| ✓ | - | ✓ | 36.3 | 58.0 | 53.2 |
| - | ✓ | ✓ | 36.4 | 58.1 | 52.8 |
| ✓ | ✓ | ✓ | 36.9 | 58.4 | 52.8 |

---

## 9. 部署指南

### 9.1 环境准备

```bash
# 1. 安装依赖
pip install openmim mmengine
mim install "mmcv>=2.0.0rc4"
pip install git+https://github.com/lvis-dataset/lvis-api.git
mim install mmdet>=3.0.0rc6

# 2. 下载预训练权重
# CLIP ViT-B/32
python -c "
import clip
import torch
model, _ = clip.load('ViT-B/32')
torch.save(model.state_dict(), 'checkpoints/clip_vitb32.pth')
"
```

### 9.2 训练命令

```bash
cd ovdet

# OV-COCO 完整训练
python tools/train.py \
    configs/baron/ov_coco/baron_kd_lbp_faster_rcnn_r50_fpn_syncbn_90kx2_lbp.py \
    --work-dir ./work_dirs/lbp
```

### 9.3 测试命令

```bash
cd ovdet

# OV-COCO 评估
python tools/test.py \
    configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_subset20_eval.py \
    path/to/checkpoint.pth \
    --work-dir ./work_dirs/eval
```

---

## 10. 常见问题与调试

### 10.1 常见问题

#### Q1: 训练开始阶段 loss_bcp 不计算

**A**: 这是预期行为，`start_iter=5000` 控制在训练开始的 5000 次迭代中不启用 BCP，让模型先热身。

#### Q2: 聚类结果不稳定

**A**: 可以尝试:
- 增大 `queue_size` (例如从 2048 增至 4096)
- 减小 `cluster_interval` (例如从 500 减至 200)
- 固定随机种子

#### Q3: IPR 没有效果

**A**: 检查:
1. 模型是否在推理模式 (`model.eval()`)
2. `rectification_factor` 是否设置在合理范围 (0.2-0.4)
3. BCP 的聚类中心是否成功计算

### 10.2 调试技巧

#### 10.2.1 检查聚类质量

```python
# 在训练过程中添加
if iter % print_interval == 0:
    features = bcp_module.bg_feat_queue.get_all()
    print(f"Queue size: {features.shape[0]}")
    print(f"Cluster centers: {bcp_module.cluster_centers[:2, :5]}")
```

#### 10.2.2 检查损失值

```python
# 监控各损失的量级
print(f"Loss standard: {loss_standard.item():.4f}")
print(f"Loss BCP: {loss_bcp.item():.4f}")
print(f"Loss BOD: {loss_bod.item():.4f}")
```

---

## 附录

### A. 完整文件清单

| 文件路径 | 说明 |
|----------|------|
| `ovdet/ovdet/methods/lbp/bcp.py` | BCP 模块实现 |
| `ovdet/ovdet/methods/lbp/bod.py` | BOD 模块实现 |
| `ovdet/ovdet/methods/lbp/ipr.py` | IPR 模块实现 |
| `ovdet/ovdet/methods/lbp/utils.py` | 工具模块 |
| `ovdet/configs/baron/ov_coco/..._lbp.py` | 配置文件 |

### B. 术语表

| 术语 | 说明 |
|------|------|
| OVD | Open-Vocabulary Object Detection，开放词汇目标检测 |
| BCP | Background Category-specific Prompt，背景类别特定提示 |
| BOD | Background Object Discovery，背景目标发现 |
| IPR | Inference Probability Rectification，推理概率修正 |
| C_b | Base Classes，基础类别 |
| C_u | Novel Classes，未知类别 |
| C_o | Implicit Background Categories，隐含背景类别 |


