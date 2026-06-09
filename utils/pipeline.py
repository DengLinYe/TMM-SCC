from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .prepare_subset import main as prepare_main
from .run_ablation import main as ablation_main
from .run_attack import main as attack_main
from .run_blackbox import main as blackbox_main
from .finetune_ve import main as finetune_main


@dataclass
class RunProfile:
    name: str
    subset: str = "main_1k"
    ablation_subset: str = "ablation_200"
    num_iters: Optional[int] = None
    attack_task: str = "all"
    attack_method: str = "all"
    blackbox_task: str = "all"
    blackbox_method: str = "all"
    blackbox_targets: Optional[Dict[str, List[str]]] = None
    ablation_iters: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    cooldown: int = 120
    gpu: int = 0
    finetune_backbone: str = "albef"
    finetune_backbones: Optional[List[str]] = None
    finetune_epochs: Optional[int] = None
    dry_run: bool = False


# 快速验证：小样本 + 低迭代，但步骤与正式实验一致（全覆盖）
QUICK_PROFILE = RunProfile(
    name="quick",
    subset="smoke_20",
    ablation_subset="smoke_20",
    num_iters=2,
    attack_task="all",
    attack_method="all",
    blackbox_task="all",
    blackbox_method="all",
    blackbox_targets={"vlr": ["tcl", "clip"], "ve": ["tcl"]},
    ablation_iters=[2, 3],
    cooldown=0,
    finetune_epochs=2,
    finetune_backbones=["albef", "tcl"],
)

FULL_PROFILE = RunProfile(
    name="full",
    subset="main_1k",
    ablation_subset="ablation_200",
    attack_task="all",
    attack_method="all",
    blackbox_task="all",
    blackbox_method="all",
    cooldown=120,
)

QUICK_STEPS = ["prepare", "finetune", "attack", "ablation", "blackbox"]
FULL_STEPS = ["prepare", "attack", "blackbox"]


def _extra_args(profile: RunProfile) -> List[str]:
    args = []
    if profile.dry_run:
        args.append("--dry-run")
    return args


def run_prepare(profile: RunProfile) -> None:
    prepare_main(["--name", profile.subset] + _extra_args(profile))


def run_attack(profile: RunProfile) -> None:
    argv = [
        "--task",
        profile.attack_task,
        "--method",
        profile.attack_method,
        "--subset",
        profile.subset,
        "--gpu",
        str(profile.gpu),
        "--cooldown",
        str(profile.cooldown),
    ] + _extra_args(profile)
    if profile.num_iters is not None:
        argv.extend(["--num-iters", str(profile.num_iters)])
    attack_main(argv)


def run_blackbox(profile: RunProfile) -> None:
    argv = [
        "--task",
        profile.blackbox_task,
        "--method",
        profile.blackbox_method,
        "--subset",
        profile.subset,
        "--gpu",
        str(profile.gpu),
    ] + _extra_args(profile)
    if profile.blackbox_targets:
        if profile.blackbox_task == "all":
            targets = []
            for key in ("vlr", "ve"):
                for t in profile.blackbox_targets.get(key, []):
                    if t not in targets:
                        targets.append(t)
        else:
            targets = profile.blackbox_targets.get(profile.blackbox_task, ["tcl"])
        if targets:
            argv.extend(["--target"] + targets)
    blackbox_main(argv)


def run_ablation(profile: RunProfile) -> None:
    argv = [
        "--subset",
        profile.ablation_subset,
        "--gpu",
        str(profile.gpu),
        "--cooldown",
        str(profile.cooldown),
        "--iters",
    ] + [str(i) for i in profile.ablation_iters] + _extra_args(profile)
    ablation_main(argv)


def run_finetune(profile: RunProfile) -> None:
    backbones = profile.finetune_backbones or [profile.finetune_backbone]
    for backbone in backbones:
        argv = [
            "--backbone",
            backbone,
            "--subset",
            profile.subset,
            "--gpu",
            str(profile.gpu),
        ] + _extra_args(profile)
        if profile.finetune_epochs is not None:
            argv.extend(["--epochs", str(profile.finetune_epochs)])
        finetune_main(argv)


STEP_RUNNERS = {
    "prepare": run_prepare,
    "attack": run_attack,
    "blackbox": run_blackbox,
    "ablation": run_ablation,
    "finetune": run_finetune,
}


def run_pipeline(steps: List[str], profile: RunProfile) -> None:
    for step in steps:
        if step not in STEP_RUNNERS:
            raise ValueError(f"未知步骤: {step}")
        print(f"\n{'#' * 60}\n# {profile.name} / {step}\n{'#' * 60}")
        STEP_RUNNERS[step](profile)
    print("\n[+] 流水线执行完毕")


def run_quick(gpu: int = 0, dry_run: bool = False) -> None:
    profile = replace(QUICK_PROFILE, gpu=gpu, dry_run=dry_run)
    run_pipeline(QUICK_STEPS, profile)


def run_full(gpu: int = 0, dry_run: bool = False, with_ablation: bool = False) -> None:
    profile = replace(FULL_PROFILE, gpu=gpu, dry_run=dry_run)
    steps = list(FULL_STEPS)
    if with_ablation:
        steps.append("ablation")
    run_pipeline(steps, profile)


# 兼容旧名
run_smoke = run_quick

PROFILES = {"quick": QUICK_PROFILE, "smoke": QUICK_PROFILE, "full": FULL_PROFILE}
