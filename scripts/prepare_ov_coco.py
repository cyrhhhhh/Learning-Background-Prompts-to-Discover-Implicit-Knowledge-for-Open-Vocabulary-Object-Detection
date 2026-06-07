from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbp_data.coco_lvis import filter_coco_like_annotations, load_json, save_json
from lbp_data.splits import COCO_BASE_CLASSES, COCO_NOVEL_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OV-COCO 48-base/17-novel annotation files."
    )
    parser.add_argument("--coco-root", default="data/raw/coco")
    parser.add_argument("--out-dir", default="data/processed/ov_coco/annotations")
    parser.add_argument(
        "--keep-empty-images",
        action="store_true",
        help="Keep images even when no annotation remains after filtering.",
    )
    return parser.parse_args()


def build_outputs(annotation: dict, keep_empty_images: bool) -> dict[str, dict]:
    split_by_name = {name: "base" for name in COCO_BASE_CLASSES}
    split_by_name.update({name: "novel" for name in COCO_NOVEL_CLASSES})
    split_by_id = {
        category["id"]: split_by_name[category["name"]]
        for category in annotation.get("categories", [])
        if category["name"] in split_by_name
    }

    return {
        "b": filter_coco_like_annotations(
            copy.deepcopy(annotation), split_by_id, {"base"}, keep_empty_images
        ),
        "t": filter_coco_like_annotations(
            copy.deepcopy(annotation), split_by_id, {"novel"}, keep_empty_images
        ),
        "all": filter_coco_like_annotations(
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
    coco_root = (ROOT / args.coco_root).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {
        "split": {
            "base_classes": len(COCO_BASE_CLASSES),
            "novel_classes": len(COCO_NOVEL_CLASSES),
        },
        "files": {},
    }

    for year_split in ("train2017", "val2017"):
        src = coco_root / "annotations" / f"instances_{year_split}.json"
        annotation = load_json(src)
        outputs = build_outputs(annotation, args.keep_empty_images)
        for suffix, data in outputs.items():
            dst = out_dir / f"ovd_ins_{year_split}_{suffix}.json"
            save_json(data, dst)
        summary["files"][year_split] = summarize(outputs)

    save_json(summary, out_dir / "split_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

