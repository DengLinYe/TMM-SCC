import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .config import DATA, DATA_PATHS, SUBSET_PRESETS, SubsetSpec


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
    flickr_out = DATA / "flickr30k" / f"flickr30k_test_{spec.name}.json"
    ve_out = DATA / "snli-ve" / f"ve_test_{spec.name}.json"

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

    meta_path = DATA / "subsets" / f"{spec.name}_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    return meta


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

    meta = create_subset(spec)
    print(f"[+] 子集 {spec.name} 已生成")
    print(f"    VLR: {meta['vlr']}")
    print(f"    VE:  {meta['ve']}")
    print(f"    meta: data/subsets/{spec.name}_meta.json")
