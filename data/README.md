# Data Layout

This directory is intentionally split by data lifecycle.

```text
data/
  raw/
    coco/
      train2017/
      val2017/
      annotations/
        instances_train2017.json
        instances_val2017.json
    lvis/
      annotations/
        lvis_v1_train.json
        lvis_v1_val.json
      train2017 -> ../coco/train2017
      val2017   -> ../coco/val2017
    voc2007/
      VOCdevkit/
        VOC2007/
    objects365v2/
      val/
        zhiyuan_objv2_val.json
    _downloads/
  processed/
    ov_coco/annotations/
    ov_lvis/annotations/
  interim/
    background/
      ov_coco/
        proposals/
        clip_features/
        clusters/
        pseudo_labels/
      ov_lvis/
        proposals/
        clip_features/
        clusters/
        pseudo_labels/
  external/
    checkpoints/
      clip/
      soco/
      baron/
```

Keep downloaded archives and extracted raw datasets under `data/raw`.
Generated open-vocabulary annotation files belong under `data/processed`.
LBP-specific background artifacts belong under `data/interim/background`.

Current local status:

- COCO 2017 train/val, LVIS v1, VOC2007 test, and Objects365 v2 val are already under `data/raw`.
- Project-format OV-COCO and OV-LVIS split annotations are already under `data/processed`.
- Compatibility annotations generated during the first data download pass are kept under `data/processed/*/annotations/compat_original_names`.
- Detailed counts and dataset notes are in `docs/LBP_datasets.md`.
