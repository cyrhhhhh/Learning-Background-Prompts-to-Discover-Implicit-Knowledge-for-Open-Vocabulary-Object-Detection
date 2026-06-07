# 学习背景提示 (LBP)

## 项目简介

这是 **"Learning Background Prompts to Discover Implicit Knowledge for Open-Vocabulary Object Detection"** 的官方实现，已被 **CVPR 2024** 录用。

LBP 是一个用于开放词汇目标检测（Open-Vocabulary Object Detection, OVD）的创新框架，通过学习背景提示来发现背景建议中的隐含知识，显著提升对基础类别和未知类别的检测性能。

## 核心特性

- **🚀 领先性能**: 在 OV-COCO 和 OV-LVIS 基准数据集上超越现有方法
- **🔧 三大创新模块**: 
  - BCP（背景类别特定提示）
  - BOD（背景目标发现）
  - IPR（推理概率修正）
- **🎯 无需先验知识**: 无需未知类别名称即可发现隐含知识

## 方法概述
<img width="1362" height="724" alt="image" src="https://github.com/user-attachments/assets/5ce3c885-3b38-4414-9cfc-a778d94c0ce3" />


```
输入图像
    │
    ▼
┌─────────────────────┐
│   主干网络 (R-50)   │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐  ┌─────────┐
│ RPN     │  │ RoI     │
│ 头      │  │ 头      │
└────┬────┘  └────┬────┘
     │             │
     └──────┬──────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
┌─────────────────────────┐
│  BCP 模块              │──┐
│  (背景提示学习)         │  │
└─────────────────────────┘  │
     │                       │
     ▼                       │
┌─────────────────────────┐  │
│  BOD 模块              │  │ 
│  (目标发现)            │  │
└─────────────────────────┘  │
     │                       │
     ▼                       │
┌─────────────────────────┐  │
│  IPR 模块              │◄─┘
│  (概率修正)            │
└─────────────────────────┘
     │
     ▼
检测输出
```

## 性能表现

### OV-COCO 数据集结果 (AP₅₀)

| 方法           | 未知 AP₅₀ | 基础 AP₅₀ | 总体 AP₅₀ |
| -------------- | --------- | --------- | --------- |
| BARON          | 35.8      | 58.2      | 52.3      |
| **LBP (本文)** | **37.8**  | **58.7**  | **53.2**  |
| **LBP (复现)** | **36.9**  | **58.4**  | **53.8**  |

### OV-LVIS 数据集结果 (AP)

| 方法           | 稀有 APᵣ | 常见 APᶜ | 频繁 APᶠ | 总体 AP  |
| -------------- | -------- | -------- | -------- | -------- |
| BARON          | 23.2     | 29.3     | 32.5     | 29.5     |
| **LBP (本文)** | **24.1** | **29.5** | **32.8** | **29.9** |
| **LBP (复现)** | **23.8** | **29.4** | **32.7** | **29.8** |

## 安装指南

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+ (推荐)
- MMDetection 3.x, MMEngine, MMCV-full

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/LBP.git
cd LBP

# 安装依赖
pip install openmim mmengine
mim install "mmcv>=2.0.0rc4"
pip install git+https://github.com/lvis-dataset/lvis-api.git
mim install mmdet>=3.0.0rc6

# 安装其他依赖
pip install -r requirements.txt
```

### 下载预训练权重

```bash
# CLIP ViT-B/32
python -c "
import clip
import torch
model, _ = clip.load('ViT-B/32')
torch.save(model.state_dict(), 'checkpoints/clip_vitb32.pth')
"
```

## 快速开始

### 模型训练

```bash
# 在 OV-COCO 上完整训练
cd ovdet
python tools/train.py \
    configs/baron/ov_coco/baron_kd_lbp_faster_rcnn_r50_fpn_syncbn_90kx2_lbp.py \
    --work-dir ./work_dirs/lbp

# 或使用提供的脚本
cd scripts
bash train_ovdet_lbp_kd.sh
```

### 模型测试

```bash
# 在 OV-COCO 上评估
cd ovdet
python tools/test.py \
    configs/baron/ov_coco/baron_kd_lbp_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_subset20_eval.py \
    path/to/checkpoint.pth \
    --work-dir ./work_dirs/eval
```

## 项目结构

```
LBP/
├── ovdet/                          # 主要代码目录
│   ├── configs/                    # 配置文件
│   │   └── baron/                 # 方法配置
│   │       └── ov_coco/          # OV-COCO 配置
│   │       └── ov_lvis/          # OV-LVIS 配置
│   ├── ovdet/
│   │   ├── methods/               # 核心方法
│   │   │   ├── lbp/             # LBP 实现
│   │   │   │   ├── bcp.py       # BCP 模块
│   │   │   │   ├── bod.py       # BOD 模块
│   │   │   │   └── ipr.py       # IPR 模块
│   │   │   └── baron/           # BARON 基线
│   │   ├── models/               # 模型定义
│   │   │   └── roi_heads/       # RoI 头
│   │   └── vlms/                 # CLIP 模型
│   └── tools/                     # 训练和测试脚本
├── scripts/                        # 辅助脚本
├── data/                           # 数据目录
├── checkpoints/                    # 模型权重
├── CODE_WIKI_LBP.md              # 详细技术文档
└── README.md
```

## 核心模块说明

### BCP（背景类别特定提示）

- 从背景建议中发现和表示底层类别
- 使用 K-means 聚类识别隐含类别
- 通过提出的 L<sub>bcp</sub> 和 L<sub>rlx</sub> 损失函数学习类别特定提示

### BOD（背景目标发现）

- 从背景建议中挖掘隐含目标知识
- 使用聚类中心作为伪标签进行训练
- 缓解模型对基础类别的过拟合问题

### IPR（推理概率修正）

- 在推理阶段修正分类概率
- 解决背景类别与未知类别之间的语义重叠问题
- 基于论文中提出的公式 (18) 实现

