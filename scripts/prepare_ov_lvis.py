from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbp_data.coco_lvis import filter_coco_like_annotations, load_json, save_json


BASE_FREQ = {"c", "f"}
NOVEL_FREQ = {"r"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OV-LVIS base/rare annotation files from LVIS v1 JSON."
    )
    parser.add_argument("--lvis-root", default="data/raw/lvis")
    parser.add_argument("--out-dir", default="data/processed/ov_lvis/annotations")
    parser.add_argument(
        "--keep-empty-images",
        action="store_true",
        help="Keep images even when no annotation remains after filtering.",
    )
    return parser.parse_args()


def split_by_category_id(annotation: dict) -> dict[int, str]:
    split_by_id: dict[int, str] = {}
    for category in annotation.get("categories", []):
        frequency = category.get("frequency")
        if frequency in BASE_FREQ:
            split_by_id[category["id"]] = "base"
        elif frequency in NOVEL_FREQ:
            split_by_id[category["id"]] = "novel"
    return split_by_id


def build_outputs(annotation: dict, keep_empty_images: bool) -> dict[str, dict]:
    split_by_id = split_by_category_id(annotation)
    return {
        "base": filter_coco_like_annotations(
            copy.deepcopy(annotation), split_by_id, {"base"}, keep_empty_images
        ),
        "rare": filter_coco_like_annotations(
            copy.deepcopy(annotation), split_by_id, {"novel"}, keep_empty_images
        ),
        "all_ovd": filter_coco_like_annotations(
            copy.deepcopy(annotation), split_by_id, {"base", "novel"}, keep_empty_images
        ),
    }


def summarize(outputs: dict[str, dict]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "images": len(data.get("images", [])),
            "annotations": len(data.get("annotations", [])),
            "categories": len(data.get("categories", [])),
        }
        for name, data in outputs.items()
    }


def main() -> None:
    args = parse_args()
    lvis_root = (ROOT / args.lvis_root).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {"files": {}}

    for year_split in ("train", "val"):
        src = lvis_root / "annotations" / f"lvis_v1_{year_split}.json"
        annotation = load_json(src)
        outputs = build_outputs(annotation, args.keep_empty_images)
        for suffix, data in outputs.items():
            dst = out_dir / f"lvis_v1_{year_split}_{suffix}.json"
            save_json(data, dst)
        summary["files"][year_split] = summarize(outputs)

    save_json(summary, out_dir / "split_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

