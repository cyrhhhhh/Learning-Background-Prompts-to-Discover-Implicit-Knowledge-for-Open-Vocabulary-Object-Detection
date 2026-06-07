import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"

COCO_BASE_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "train", "truck",
    "boat", "bench", "bird", "horse", "sheep", "bear", "zebra",
    "giraffe", "backpack", "handbag", "suitcase", "frisbee",
    "skis", "kite", "surfboard", "bottle", "fork", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "pizza", "donut", "chair", "bed", "toilet", "tv", "laptop",
    "mouse", "remote", "microwave", "oven", "toaster",
    "refrigerator", "book", "clock", "vase", "toothbrush",
)

COCO_NOVEL_CLASSES = (
    "airplane", "bus", "cat", "dog", "cow", "elephant",
    "umbrella", "tie", "snowboard", "skateboard", "cup", "knife",
    "cake", "couch", "keyboard", "sink", "scissors",
)


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def prepare_coco():
    ann_dir = ROOT / "coco" / "annotations"
    train_path = ann_dir / "instances_train2017.json"
    val_path = ann_dir / "instances_val2017.json"
    if not train_path.exists() or not val_path.exists():
        return {"status": "missing", "reason": "COCO train/val annotations not found"}

    train_all = read_json(train_path)
    val_all = read_json(val_path)

    class_id_to_split = {}
    for item in train_all["categories"]:
        if item["name"] in COCO_BASE_CLASSES:
            class_id_to_split[item["id"]] = "seen"
        elif item["name"] in COCO_NOVEL_CLASSES:
            class_id_to_split[item["id"]] = "unseen"

    def filter_annotation(anno_dict, split_names):
        out = deepcopy(anno_dict)
        split_names = set(split_names)
        filtered_categories = []
        for item in out["categories"]:
            split = class_id_to_split.get(item["id"])
            if split in split_names:
                item["split"] = split
                filtered_categories.append(item)

        useful_image_ids = set()
        filtered_annotations = []
        for item in out["annotations"]:
            if class_id_to_split.get(item["category_id"]) in split_names:
                filtered_annotations.append(item)
                useful_image_ids.add(item["image_id"])

        out["categories"] = filtered_categories
        out["annotations"] = filtered_annotations
        out["images"] = [item for item in out["images"] if item["id"] in useful_image_ids]
        return out

    train_seen = filter_annotation(train_all, ["seen"])
    val_all_ovd = filter_annotation(val_all, ["seen", "unseen"])
    write_json(ann_dir / "instances_train2017_seen_2.json", train_seen)
    write_json(ann_dir / "instances_val2017_all_2.json", val_all_ovd)

    excluded = sorted(
        cat["name"] for cat in train_all["categories"]
        if cat["name"] not in set(COCO_BASE_CLASSES) | set(COCO_NOVEL_CLASSES)
    )
    return {
        "status": "ok",
        "base_classes": len(COCO_BASE_CLASSES),
        "novel_classes": len(COCO_NOVEL_CLASSES),
        "excluded_classes": excluded,
        "train_seen_images": len(train_seen["images"]),
        "train_seen_annotations": len(train_seen["annotations"]),
        "val_all_images": len(val_all_ovd["images"]),
        "val_all_annotations": len(val_all_ovd["annotations"]),
        "outputs": [
            str(ann_dir / "instances_train2017_seen_2.json"),
            str(ann_dir / "instances_val2017_all_2.json"),
        ],
    }


def valid_lvis_ann(ann, img):
    if ann.get("ignore", False) or ann.get("iscrowd", False):
        return False
    x1, y1, w, h = ann["bbox"]
    inter_w = max(0, min(x1 + w, img["width"]) - max(x1, 0))
    inter_h = max(0, min(y1 + h, img["height"]) - max(y1, 0))
    return inter_w * inter_h > 0 and ann.get("area", 0) > 0 and w >= 1 and h >= 1


def lvis_filename(img):
    coco_url = img.get("coco_url")
    if coco_url:
        return coco_url.replace("http://images.cocodataset.org/", "")
    return img.get("file_name", "")


def prepare_lvis():
    ann_dir = ROOT / "lvis" / "annotations"
    train_path = ann_dir / "lvis_v1_train.json"
    val_path = ann_dir / "lvis_v1_val.json"
    if not train_path.exists() or not val_path.exists():
        return {"status": "missing", "reason": "LVIS v1 train/val annotations not found"}

    train = read_json(train_path)
    val = read_json(val_path)

    id_to_cat = {cat["id"]: cat for cat in val["categories"]}
    id_to_name = {cat_id: cat["name"] for cat_id, cat in id_to_cat.items()}
    rare_cat_ids = {cat["id"] for cat in val["categories"] if cat.get("frequency") == "r"}
    base_cat_ids = {cat["id"] for cat in val["categories"] if cat.get("frequency") != "r"}
    base_zero_ids = {cat_id - 1 for cat_id in base_cat_ids}

    label_map = {str(cat["id"] - 1): cat["name"] for cat in val["categories"]}
    label_map_norare = {
        str(cat["id"] - 1): cat["name"]
        for cat in val["categories"] if cat["id"] in base_cat_ids
    }
    write_json(ann_dir / "lvis_v1_label_map.json", label_map)
    write_json(ann_dir / "lvis_v1_label_map_norare.json", label_map_norare)

    train_norare = deepcopy(train)
    useful_image_ids = set()
    train_norare["annotations"] = []
    for ann in train["annotations"]:
        if ann["category_id"] in base_cat_ids:
            train_norare["annotations"].append(ann)
            useful_image_ids.add(ann["image_id"])
    train_norare["images"] = [img for img in train["images"] if img["id"] in useful_image_ids]
    train_norare["categories"] = [cat for cat in train["categories"] if cat["id"] in base_cat_ids]
    write_json(ann_dir / "lvis_v1_train_norare.json", train_norare)

    def to_odvg_rows(keep_base_only):
        anns_by_image = defaultdict(list)
        img_by_id = {img["id"]: img for img in train["images"]}
        for ann in train["annotations"]:
            if keep_base_only and ann["category_id"] not in base_cat_ids:
                continue
            img = img_by_id.get(ann["image_id"])
            if img is None or not valid_lvis_ann(ann, img):
                continue
            x1, y1, w, h = ann["bbox"]
            label = ann["category_id"] - 1
            anns_by_image[ann["image_id"]].append({
                "bbox": [x1, y1, x1 + w, y1 + h],
                "label": label,
                "category": id_to_name[ann["category_id"]],
            })

        rows = []
        for img in train["images"]:
            instances = anns_by_image.get(img["id"], [])
            if keep_base_only and not instances:
                continue
            rows.append({
                "filename": lvis_filename(img),
                "height": img["height"],
                "width": img["width"],
                "detection": {"instances": instances},
            })
        return rows

    odvg_all = to_odvg_rows(False)
    odvg_norare = to_odvg_rows(True)
    write_jsonl(ann_dir / "lvis_v1_train_od.json", odvg_all)
    write_jsonl(ann_dir / "lvis_v1_train_od_norare.json", odvg_norare)

    return {
        "status": "ok",
        "all_classes": len(val["categories"]),
        "rare_novel_classes": len(rare_cat_ids),
        "base_common_frequent_classes": len(base_cat_ids),
        "base_zero_index_count": len(base_zero_ids),
        "train_norare_images": len(train_norare["images"]),
        "train_norare_annotations": len(train_norare["annotations"]),
        "odvg_train_rows": len(odvg_all),
        "odvg_train_norare_rows": len(odvg_norare),
        "outputs": [
            str(ann_dir / "lvis_v1_train_norare.json"),
            str(ann_dir / "lvis_v1_label_map.json"),
            str(ann_dir / "lvis_v1_label_map_norare.json"),
            str(ann_dir / "lvis_v1_train_od.json"),
            str(ann_dir / "lvis_v1_train_od_norare.json"),
        ],
    }


def main():
    stats = {
        "dataset_root": str(ROOT),
        "coco_ovd": prepare_coco(),
        "lvis_ovd": prepare_lvis(),
    }
    out = ROOT / "lbp_dataset_prepare_stats.json"
    write_json(out, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
