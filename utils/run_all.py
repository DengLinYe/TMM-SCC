import argparse
import sys

from utils.prepare_subset import main as prepare_main
from utils.run_attack import main as attack_main
from utils.run_ablation import main as ablation_main
from utils.run_blackbox import main as blackbox_main
from utils.finetune_ve import main as finetune_main

STEPS = {
    "prepare": prepare_main,
    "attack": attack_main,
    "ablation": ablation_main,
    "blackbox": blackbox_main,
    "finetune": finetune_main,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="一键运行实验流水线（可单独或组合执行各步骤）"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEPS.keys()) + ["all"],
        default=["all"],
        help="prepare / attack / ablation / blackbox / finetune / all",
    )
    parser.add_argument("--subset", default="main_1k")
    parser.add_argument("--ablation-subset", default="ablation_200")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)

    steps = ["prepare", "attack", "blackbox"] if "all" in args.steps else args.steps
    if args.skip_ablation and "ablation" in steps:
        steps.remove("ablation")

    extra_args = extra
    if args.dry_run and "--dry-run" not in extra_args:
        extra_args = extra_args + ["--dry-run"]

    for step in steps:
        print(f"\n{'#' * 60}\n# STEP: {step}\n{'#' * 60}")
        step_argv = list(extra_args)
        if step == "prepare":
            step_argv = ["--name", args.subset] + step_argv
        elif step == "attack":
            step_argv = ["--subset", args.subset, "--gpu", str(args.gpu)] + step_argv
        elif step == "ablation":
            step_argv = [
                "--subset",
                args.ablation_subset,
                "--gpu",
                str(args.gpu),
            ] + step_argv
        elif step == "blackbox":
            step_argv = ["--subset", args.subset] + step_argv
        STEPS[step](step_argv)

    print("\n[+] 流水线执行完毕")


if __name__ == "__main__":
    main()
