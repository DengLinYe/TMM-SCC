import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .config import DATA, DATA_PATHS, SUBSET_PRESETS, SubsetSpec, rel_path, ve_finetune_counts
from .log import info, ok, require_file


def sample_vlr_by_image(
    source: Path, target: Path, image_count: int, seed: int
) -> dict:
    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_to_entries = {}
    for entry in data:
        img_id = entry.get("image") or entry.get("image_id")
        img_to_entries.setdefault(img_id, []).append(entry)

    all_images = list(img_to_entries.keys())
    rng = random.Random(seed)
    sampled = rng.sample(all_images, min(image_count, len(all_images)))

    subset = []
    for img_id in sampled:
        subset.extend(img_to_entries[img_id])

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=4, ensure_ascii=False)

    return {
        "unique_images": len(sampled),
        "total_entries": len(subset),
    }


def sample_ve_by_entry(
    source: Path, target: Path, entry_count: int, seed: int
) -> dict:
    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    sampled = rng.sample(data, min(entry_count, len(data)))

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=4, ensure_ascii=False)

    labels = [e["label"] for e in sampled if "label" in e]
    return {
        "total_entries": len(sampled),
        "label_distribution": dict(Counter(labels)),
    }


def create_subset(spec: SubsetSpec) -> dict:
    require_file(DATA_PATHS["flickr_test_full"], "Flickr 测试标注", module="prepare")
    require_file(DATA_PATHS["ve_test_full"], "SNLI-VE 测试标注", module="prepare")

    flickr_out = DATA / "flickr30k" / f"flickr30k_test_{spec.name}.json"
    ve_out = DATA / "snli-ve" / f"ve_test_{spec.name}.json"
    ve_splits = ve_finetune_counts(spec)

    meta = {
        "name": spec.name,
        "seed": spec.seed,
        "vlr": sample_vlr_by_image(
            DATA_PATHS["flickr_test_full"],
            flickr_out,
            spec.vlr_image_count,
            spec.seed,
        ),
        "ve": sample_ve_by_entry(
            DATA_PATHS["ve_test_full"],
            ve_out,
            spec.ve_entry_count,
            spec.seed,
        ),
    }

    if ve_splits:
        require_file(DATA_PATHS["ve_train_full"], "SNLI-VE 训练标注", module="prepare")
        require_file(DATA_PATHS["ve_dev_full"], "SNLI-VE 验证标注", module="prepare")
        ve_train_out = DATA / "snli-ve" / f"ve_train_{spec.name}.json"
        ve_dev_out = DATA / "snli-ve" / f"ve_dev_{spec.name}.json"
        meta["ve_finetune"] = {
            "train": sample_ve_by_entry(
                DATA_PATHS["ve_train_full"],
                ve_train_out,
                ve_splits["train"],
                spec.seed + 1,
            ),
            "dev": sample_ve_by_entry(
                DATA_PATHS["ve_dev_full"],
                ve_dev_out,
                ve_splits["dev"],
                spec.seed + 2,
            ),
        }

    meta_path = DATA / "subsets" / f"{spec.name}_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    return meta


def describe_subset(spec: SubsetSpec) -> dict:
    plan = {
        "name": spec.name,
        "seed": spec.seed,
        "vlr_images": spec.vlr_image_count,
        "ve_entries": spec.ve_entry_count,
        "vlr_out": DATA / "flickr30k" / f"flickr30k_test_{spec.name}.json",
        "ve_out": DATA / "snli-ve" / f"ve_test_{spec.name}.json",
        "meta_out": DATA / "subsets" / f"{spec.name}_meta.json",
    }
    ve_splits = ve_finetune_counts(spec)
    if ve_splits:
        plan["ve_train_out"] = DATA / "snli-ve" / f"ve_train_{spec.name}.json"
        plan["ve_dev_out"] = DATA / "snli-ve" / f"ve_dev_{spec.name}.json"
        plan["ve_train_count"] = ve_splits["train"]
        plan["ve_dev_count"] = ve_splits["dev"]
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成 VLR/VE 标注子集")
    parser.add_argument(
        "--name",
        default="main_1k",
        help="子集名称，如 main_1k / ablation_200",
    )
    parser.add_argument("--vlr-images", type=int, default=None)
    parser.add_argument("--ve-entries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.name in SUBSET_PRESETS:
        preset = SUBSET_PRESETS[args.name]
        spec = SubsetSpec(
            name=args.name,
            vlr_image_count=args.vlr_images or preset.vlr_image_count,
            ve_entry_count=args.ve_entries or preset.ve_entry_count,
            seed=args.seed,
        )
    else:
        if args.vlr_images is None or args.ve_entries is None:
            parser.error("自定义子集需同时指定 --vlr-images 和 --ve-entries")
        spec = SubsetSpec(
            name=args.name,
            vlr_image_count=args.vlr_images,
            ve_entry_count=args.ve_entries,
            seed=args.seed,
        )

    if args.dry_run:
        plan = describe_subset(spec)
        info(
            f"采样 VLR {plan['vlr_images']} 图 -> {rel_path(plan['vlr_out'])}",
            module="prepare",
        )
        info(
            f"采样 VE {plan['ve_entries']} 条 -> {rel_path(plan['ve_out'])}",
            module="prepare",
        )
        if "ve_train_out" in plan:
            info(
                f"采样 VE train {plan['ve_train_count']} 条 -> {rel_path(plan['ve_train_out'])}",
                module="prepare",
            )
            info(
                f"采样 VE dev {plan['ve_dev_count']} 条 -> {rel_path(plan['ve_dev_out'])}",
                module="prepare",
            )
        info(f"meta -> {rel_path(plan['meta_out'])}", module="prepare")
        ok(f"dry-run: 子集 {spec.name} 配置就绪", module="prepare")
        return

    meta = create_subset(spec)
    ok(f"子集 {spec.name} 已生成", module="prepare")
    info(f"VLR: {meta['vlr']}", module="prepare")
    info(f"VE:  {meta['ve']}", module="prepare")
    info(f"meta: data/subsets/{spec.name}_meta.json", module="prepare")
