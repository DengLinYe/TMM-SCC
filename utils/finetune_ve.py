import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML

from .config import (
    CHECKPOINT_MAP,
    DATA,
    DATA_PATHS,
    FINETUNE,
    ROOT,
    finetune_output_dir,
    resolve_checkpoint,
    third_party_rel,
)
from .log import cmdline, die, info, ok, require_file, run_subprocess, skip, warn
from .output import reset_output_dir

THIRD_PARTY = {
    "albef": ROOT / "third_party" / "albef",
    "tcl": ROOT / "third_party" / "tcl",
}


def ve_ann_path(split: str, subset: str) -> Path:
    keyed = DATA / "snli-ve" / f"ve_{split}_{subset}.json"
    if keyed.exists():
        return keyed
    return DATA / "snli-ve" / f"ve_{split}.json"


def ve_checkpoint_ready(backbone: str) -> bool:
    ckpt = CHECKPOINT_MAP["ve"].get(backbone)
    return ckpt is not None and ckpt.is_file() and ckpt.stat().st_size > 0


def finetune_init_ckpt(backbone: str) -> Optional[Path]:
    """微调始终从 VLR 权重 warm-start，保证每次运行路径一致。"""
    vlr_ckpt = resolve_checkpoint("vlr", backbone)
    if vlr_ckpt and vlr_ckpt.exists():
        return vlr_ckpt
    return None


def write_finetune_config(
    backbone: str,
    subset: str,
    repo: Path,
    out_dir: Path,
    epochs: int = None,
) -> Path:
    yaml = YAML()
    yaml.preserve_quotes = True
    template = repo / "configs" / "VE.yaml"
    with open(template, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    config["train_file"] = third_party_rel(repo, ve_ann_path("train", subset))
    config["val_file"] = third_party_rel(repo, ve_ann_path("dev", subset))
    config["test_file"] = third_party_rel(repo, ve_ann_path("test", subset))
    config["image_root"] = third_party_rel(repo, DATA_PATHS["flickr_image_root"]) + "/"
    config["bert_config"] = third_party_rel(repo, repo / "configs" / "config_bert.json")
    config["num_workers"] = FINETUNE.num_workers
    config["distill"] = FINETUNE.distill

    if epochs is not None and "schedular" in config:
        config["schedular"]["epochs"] = epochs

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return cfg_path


def copy_best_checkpoint(backbone: str, best_path: Path):
    dest = CHECKPOINT_MAP["ve"][backbone]
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, dest)
    ok(f"best 权重已复制到 {dest}", module="finetune")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VE 任务微调（ALBEF / TCL）")
    parser.add_argument("--backbone", choices=["albef", "tcl"], required=True)
    parser.add_argument("--subset", default="main_1k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--copy-best",
        default=None,
        help="指定 best checkpoint 路径并复制到 checkpoints/ve/",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="checkpoints/ve 已有权重时跳过微调",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略 --skip-if-exists，强制重新微调",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo = THIRD_PARTY[args.backbone]
    if not repo.exists():
        die(f"未找到 {repo}", module="finetune")

    if args.skip_if_exists and not args.force and ve_checkpoint_ready(args.backbone):
        skip(
            f"{args.backbone} VE 权重已存在: {CHECKPOINT_MAP['ve'][args.backbone]}",
            module="finetune",
        )
        return 0

    for split in ("train", "dev", "test"):
        ann = ve_ann_path(split, args.subset)
        if not args.dry_run:
            require_file(ann, f"VE {split} 标注 ({args.subset})", module="finetune")

    init_ckpt = finetune_init_ckpt(args.backbone)
    if init_ckpt is None and not args.dry_run:
        die(f"未找到 VLR 初始化权重 ({args.backbone})", module="finetune")

    out_dir = finetune_output_dir(args.backbone)
    if not args.dry_run and not args.copy_best:
        reset_output_dir(out_dir)
    cfg_path = write_finetune_config(
        args.backbone,
        args.subset,
        repo,
        out_dir,
        epochs=args.epochs,
    )

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
    if init_ckpt:
        cmd.extend(["--checkpoint", str(init_ckpt.resolve())])

    info(f"微调: {args.backbone}", module="finetune")
    info(f"配置: {cfg_path}", module="finetune")
    info(f"输出: {out_dir}", module="finetune")
    if init_ckpt:
        info(f"初始化权重 (VLR): {init_ckpt}", module="finetune")
    info(
        "VLR→VE warm-start 会部分加载权重（itm_head/queue 等检索头忽略属正常）",
        module="finetune",
    )
    info("distill=False（固定 VLR warm-start 路径）", module="finetune")
    if args.dry_run:
        cmdline(cmd)
        return 0

    if args.copy_best:
        copy_best_checkpoint(args.backbone, Path(args.copy_best))
        return 0

    rc = run_subprocess(
        cmd,
        cwd=repo,
        module="finetune",
        label=f"VE 微调 {args.backbone}",
    )
    if rc != 0:
        return rc

    best_ckpt = out_dir / "checkpoint_best_2.pth"
    if best_ckpt.exists():
        copy_best_checkpoint(args.backbone, best_ckpt)
    else:
        warn(f"未找到 {best_ckpt}，请手动 --copy-best", module="finetune")
        return 1

    from .config import attack_config_for
    from .clean_eval import refresh_ve_clean_after_finetune

    attack = attack_config_for(gpu=args.gpu)
    refresh_ve_clean_after_finetune(
        args.backbone, args.subset, attack, dry_run=args.dry_run
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
