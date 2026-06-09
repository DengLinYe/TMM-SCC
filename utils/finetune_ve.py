import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ruamel.yaml import YAML

from .config import CHECKPOINTS, DATA, ROOT, finetune_output_dir

THIRD_PARTY = {
    "albef": ROOT / "third_party" / "albef",
    "tcl": ROOT / "third_party" / "tcl",
}


def write_finetune_config(backbone: str, subset: str, out_dir: Path) -> Path:
    yaml = YAML()
    yaml.preserve_quotes = True
    template = THIRD_PARTY[backbone] / "configs" / "VE.yaml"
    with open(template, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    test_ann = DATA / "snli-ve" / f"ve_test_{subset}.json"
    if not test_ann.exists():
        test_ann = DATA / "snli-ve" / "ve_test.json"

    config["train_file"] = f"../../data/snli-ve/ve_train.json"
    config["val_file"] = f"../../data/snli-ve/ve_dev.json"
    config["test_file"] = f"../../data/snli-ve/{test_ann.name}"
    config["image_root"] = "../../data/flickr30k/flickr30k-images/"

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return cfg_path


def copy_best_checkpoint(backbone: str, best_path: Path):
    dest = CHECKPOINTS / "VE" / f"{backbone}_ve_snli_ve.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, dest)
    print(f"[+] best 权重已复制到 {dest}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="VE 任务微调（ALBEF / TCL）")
    parser.add_argument("--backbone", choices=["albef", "tcl"], required=True)
    parser.add_argument("--subset", default="main_1k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--copy-best",
        default=None,
        help="指定 best checkpoint 路径并复制到 checkpoints/VE/",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo = THIRD_PARTY[args.backbone]
    if not repo.exists():
        print(f"[!] 未找到 {repo}")
        sys.exit(1)

    out_dir = finetune_output_dir(args.backbone)
    cfg_path = write_finetune_config(args.backbone, args.subset, out_dir)

    cmd = [
        sys.executable,
        "VE.py",
        "--config",
        str(cfg_path.resolve()),
        "--output_dir",
        str(out_dir.resolve()),
        "--device",
        f"cuda:{args.gpu}",
    ]

    print(f"[i] 微调: {args.backbone}")
    print(f"[i] 配置: {cfg_path}")
    print(f"[i] 输出: {out_dir}")
    if args.dry_run:
        print(" ".join(cmd))
        return

    if args.copy_best:
        copy_best_checkpoint(args.backbone, Path(args.copy_best))
        return

    result = subprocess.run(cmd, cwd=str(repo))
    if result.returncode != 0:
        sys.exit(result.returncode)

    best_ckpt = out_dir / "checkpoint_best_2.pth"
    if best_ckpt.exists():
        copy_best_checkpoint(args.backbone, best_ckpt)
    else:
        print(f"[!] 未找到 {best_ckpt}，请手动 --copy-best")
