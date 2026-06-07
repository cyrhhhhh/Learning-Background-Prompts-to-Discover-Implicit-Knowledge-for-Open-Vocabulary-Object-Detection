from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_PATHS = {
    "paper": "Li_Learning_Background_Prompts_to_Discover_Implicit_Knowledge_for_Open_Vocabulary_CVPR_2024_paper.pdf",
    "compat_train_ann": "data/processed/ov_coco/annotations/compat_wusize/instances_train2017_base.json",
    "compat_base_val_ann": "data/processed/ov_coco/annotations/compat_wusize/instances_val2017_base.json",
    "compat_novel_val_ann": "data/processed/ov_coco/annotations/compat_wusize/instances_val2017_novel.json",
    "clip_state_dict": "ovdet/checkpoints/clip_vitb32.pth",
    "soco_backbone": "ovdet/checkpoints/res50_fpn_soco_star_400.pth",
    "smoke_config": "ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_smoke.py",
    "mini50_config": "ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_mini50.py",
    "mini100_config": "ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_mini100.py",
    "subset20_eval_config": "ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2_lbp_subset20_eval.py",
    "mini50_checkpoint": "outputs/ovdet_train_kd_mini50/iter_50.pth",
    "mini100_checkpoint": "outputs/ovdet_train_kd_mini100_resume/iter_100.pth",
    "official_baron_r50_fpn_checkpoint": "data/external/checkpoints/baron/r50_fpn_clip/iter_90000.pth",
    "official_baron_full_eval_json": "outputs/ovdet_official_baron_r50_fpn_full_eval/20260603_164507/20260603_164507.json",
    "handoff_report": "docs/lbp_middle_handoff_20260603.md",
    "paper_alignment_report": "docs/lbp_reproduction_alignment_20260603.md",
    "performance_report": "docs/lbp_performance_reproduction_result_20260603.md",
}

METHOD_PATTERNS = {
    "BCP": [
        r"Background Category-specific Prompt",
        r"BackgroundCategory",
        r"\bLbcp\b",
        r"\bloss_bcp\b",
        r"\bbcp\b",
    ],
    "BOD": [
        r"Background Object Discovery",
        r"\bloss_bod\b",
        r"\bbod\b",
    ],
    "IPR": [
        r"Inference Probability Rectification",
        r"Probability Rectification",
        r"\bloss_ipr\b",
        r"\bipr\b",
    ],
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def scan_method_keywords(root: Path) -> dict[str, dict[str, object]]:
    code_roots = [
        root / "ovdet" / "ovdet",
        root / "ovdet" / "configs",
    ]
    files: list[Path] = []
    for code_root in code_roots:
        if code_root.exists():
            files.extend(
                p for p in code_root.rglob("*")
                if p.suffix in {".py", ".md", ".yaml", ".yml"}
            )

    result: dict[str, dict[str, object]] = {}
    for module, patterns in METHOD_PATTERNS.items():
        hits: list[str] = []
        for file_path in files:
            text = read_text(file_path)
            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
                hits.append(str(file_path.relative_to(root)))
        result[module] = {
            "implemented_or_mentioned_in_code": bool(hits),
            "hits": hits[:20],
        }
    return result


def build_report(root: Path) -> dict[str, object]:
    paths = {
        name: {
            "path": rel_path,
            "exists": (root / rel_path).exists(),
        }
        for name, rel_path in REQUIRED_PATHS.items()
    }
    method_scan = scan_method_keywords(root)
    runnable_chain = all(
        paths[name]["exists"]
        for name in [
            "compat_train_ann",
            "clip_state_dict",
            "soco_backbone",
            "mini50_config",
            "mini100_config",
            "subset20_eval_config",
            "mini50_checkpoint",
            "mini100_checkpoint",
            "official_baron_r50_fpn_checkpoint",
            "official_baron_full_eval_json",
        ]
    )
    lbp_core_implemented = all(
        method_scan[module]["implemented_or_mentioned_in_code"]
        for module in ["BCP", "BOD", "IPR"]
    )
    return {
        "root": str(root),
        "baseline_train_eval_chain_ready": runnable_chain,
        "lbp_core_modules_detected": lbp_core_implemented,
        "paths": paths,
        "method_scan": method_scan,
        "interpretation": (
            "BARON/ovdet short training, subset evaluation, and official "
            "R50-FPN full-val baseline evaluation are ready; "
            "LBP core modules still need implementation and optimization."
        ),
    }


def print_summary(report: dict[str, object]) -> None:
    print("LBP reproduction audit")
    print(f"root: {report['root']}")
    print(f"baseline_train_eval_chain_ready: {report['baseline_train_eval_chain_ready']}")
    print(f"lbp_core_modules_detected: {report['lbp_core_modules_detected']}")
    print()
    print("Required paths:")
    paths = report["paths"]
    assert isinstance(paths, dict)
    for name, item in paths.items():
        assert isinstance(item, dict)
        status = "OK" if item["exists"] else "MISSING"
        print(f"- {status:7} {name}: {item['path']}")
    print()
    print("LBP method keyword scan:")
    method_scan = report["method_scan"]
    assert isinstance(method_scan, dict)
    for module, item in method_scan.items():
        assert isinstance(item, dict)
        status = "FOUND" if item["implemented_or_mentioned_in_code"] else "MISSING"
        print(f"- {module}: {status}")
        for hit in item["hits"]:
            print(f"  {hit}")
    print()
    print(report["interpretation"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local LBP reproduction handoff state.")
    parser.add_argument(
        "--root",
        default=".",
        help="Project root. Defaults to current working directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human summary.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = build_report(root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
