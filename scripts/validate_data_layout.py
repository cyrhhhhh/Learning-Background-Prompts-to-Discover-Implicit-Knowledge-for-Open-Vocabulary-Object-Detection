from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_RAW = [
    "data/raw/coco/annotations/instances_train2017.json",
    "data/raw/coco/annotations/instances_val2017.json",
    "data/raw/lvis/annotations/lvis_v1_train.json",
    "data/raw/lvis/annotations/lvis_v1_val.json",
]

EXPECTED_PROCESSED = [
    "data/processed/ov_coco/annotations/ovd_ins_train2017_b.json",
    "data/processed/ov_coco/annotations/ovd_ins_val2017_all.json",
    "data/processed/ov_lvis/annotations/lvis_v1_train_base.json",
    "data/processed/ov_lvis/annotations/lvis_v1_val_all_ovd.json",
]


def summarize_json(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return (
        f"images={len(data.get('images', []))}, "
        f"annotations={len(data.get('annotations', []))}, "
        f"categories={len(data.get('categories', []))}"
    )


def check(paths: list[str], label: str) -> None:
    print(f"\n{label}")
    for item in paths:
        path = ROOT / item
        if path.exists():
            suffix = summarize_json(path) if path.suffix == ".json" else "exists"
            print(f"  OK      {item} ({suffix})")
        else:
            print(f"  MISSING {item}")


def main() -> None:
    check(REQUIRED_RAW, "Raw annotations")
    check(EXPECTED_PROCESSED, "Processed annotations")

    background_root = ROOT / "data/interim/background"
    print("\nBackground cache directories")
    for item in [
        "ov_coco/proposals",
        "ov_coco/clip_features",
        "ov_coco/clusters",
        "ov_coco/pseudo_labels",
        "ov_lvis/proposals",
        "ov_lvis/clip_features",
        "ov_lvis/clusters",
        "ov_lvis/pseudo_labels",
    ]:
        path = background_root / item
        print(f"  {'OK     ' if path.exists() else 'MISSING'} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

