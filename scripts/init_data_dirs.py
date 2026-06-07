from __future__ import annotations

from pathlib import Path


DIRS = [
    "data/raw/coco/annotations",
    "data/raw/coco/train2017",
    "data/raw/coco/val2017",
    "data/raw/lvis/annotations",
    "data/processed/ov_coco/annotations",
    "data/processed/ov_lvis/annotations",
    "data/interim/background/ov_coco/proposals",
    "data/interim/background/ov_coco/clip_features",
    "data/interim/background/ov_coco/clusters",
    "data/interim/background/ov_coco/pseudo_labels",
    "data/interim/background/ov_lvis/proposals",
    "data/interim/background/ov_lvis/clip_features",
    "data/interim/background/ov_lvis/clusters",
    "data/interim/background/ov_lvis/pseudo_labels",
    "data/external/checkpoints/clip",
    "data/external/checkpoints/soco",
    "data/external/checkpoints/baron",
    "outputs/reports",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for item in DIRS:
        path = root / item
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        keep.touch(exist_ok=True)
        print(f"ok {path.relative_to(root)}")


if __name__ == "__main__":
    main()

