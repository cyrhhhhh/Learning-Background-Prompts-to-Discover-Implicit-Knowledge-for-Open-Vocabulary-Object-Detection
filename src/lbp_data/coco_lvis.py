from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)


def filter_coco_like_annotations(
    annotation: dict,
    split_by_category_id: dict[int, str],
    allowed_splits: Iterable[str],
    keep_empty_images: bool = False,
) -> dict:
    allowed = set(allowed_splits)

    categories = []
    allowed_category_ids = set()
    for category in annotation.get("categories", []):
        split = split_by_category_id.get(category["id"])
        if split in allowed:
            category = dict(category)
            category["split"] = split
            categories.append(category)
            allowed_category_ids.add(category["id"])

    annotations = []
    useful_image_ids = set()
    for item in annotation.get("annotations", []):
        if item.get("category_id") in allowed_category_ids:
            annotations.append(item)
            useful_image_ids.add(item["image_id"])

    if keep_empty_images:
        images = annotation.get("images", [])
    else:
        images = [
            item for item in annotation.get("images", []) if item["id"] in useful_image_ids
        ]

    annotation["categories"] = categories
    annotation["annotations"] = annotations
    annotation["images"] = images
    return annotation

