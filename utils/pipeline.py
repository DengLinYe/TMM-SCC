"""按步骤串联 prepare / finetune / attack / ablation / blackbox。"""

from typing import List

from .config import (
    CLOUD_ABLATION_PROFILE,
    CLOUD_MAIN_FULL_PROFILE,
    CLOUD_MAIN_NO_FINETUNE_PROFILE,
    CLOUD_TEST_FULL_PROFILE,
    PIPELINE_FULL_PROFILE,
    PIPELINE_TEST_FORCE_PROFILE,
    PIPELINE_TEST_PROFILE,
    SUBSET_PRESETS,
    RunProfile,
    apply_hardware,
    profile_with_gpu,
)
from .finetune_ve import main as finetune_main
from .log import banner, die, ok, warn
from .output import append_run_log, build_experiment_record, clean_for_cloud_main
from .prepare_subset import main as prepare_main
from .run_ablation import main as ablation_main
from .run_attack import main as attack_main
from .run_blackbox import main as blackbox_main


def _extra_args(profile: RunProfile) -> List[str]:
    args = []
    if profile.dry_run:
        args.append("--dry-run")
    return args


def run_prepare(profile: RunProfile) -> None:
    prepare_main(["--name", profile.subset] + _extra_args(profile))


def run_attack(profile: RunProfile) -> None:
    attack_main(
        [
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
        ]
        + _extra_args(profile),
        profile=profile,
    )


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
    blackbox_main(argv, profile=profile)


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
    ablation_main(argv, profile=profile)


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
        if profile.skip_finetune_if_ready and not profile.force_finetune:
            argv.append("--skip-if-exists")
        if profile.force_finetune:
            argv.append("--force")
        rc = finetune_main(argv)
        if rc != 0:
            die(f"步骤 finetune 失败: backbone={backbone} (exit {rc})", module="pipeline")


STEP_RUNNERS = {
    "prepare": run_prepare,
    "attack": run_attack,
    "blackbox": run_blackbox,
    "ablation": run_ablation,
    "finetune": run_finetune,
}


def run_pipeline(steps: List[str], profile: RunProfile) -> None:
    if profile.clean_outputs_before and not profile.dry_run:
        warn("清理 outputs/（保留 run.json 与 ablation_200）", module="pipeline")
        clean_for_cloud_main()
    if not profile.dry_run:
        spec = SUBSET_PRESETS.get(profile.subset)
        config = {
            "profile": profile.name,
            "subset": profile.subset,
            "steps": steps,
        }
        if spec:
            config["vlr_image_count"] = spec.vlr_image_count
            config["ve_entry_count"] = spec.ve_entry_count
        append_run_log(
            build_experiment_record(f"pipeline_{profile.name}", config, {})
        )
    for step in steps:
        if step not in STEP_RUNNERS:
            die(f"未知步骤: {step}", module="pipeline")
        banner(f"{profile.name} / {step}")
        STEP_RUNNERS[step](profile)
    ok("流水线执行完毕", module="pipeline")


def run_pipeline_test() -> None:
    profile = profile_with_gpu(PIPELINE_TEST_PROFILE)
    run_pipeline(profile.steps, profile)


def run_pipeline_full() -> None:
    profile = profile_with_gpu(PIPELINE_FULL_PROFILE)
    run_pipeline(profile.steps, profile)


def run_pipeline_test_force() -> None:
    profile = profile_with_gpu(PIPELINE_TEST_FORCE_PROFILE)
    run_pipeline(profile.steps, profile)


def run_cloud_test_full(preset: str = "server_24g") -> None:
    profile = apply_hardware(profile_with_gpu(CLOUD_TEST_FULL_PROFILE), preset)
    run_pipeline(profile.steps, profile)


def run_cloud_ablation(preset: str = "server_24g") -> None:
    profile = apply_hardware(profile_with_gpu(CLOUD_ABLATION_PROFILE), preset)
    run_pipeline(profile.steps, profile)


def run_cloud_main_full(preset: str = "server_24g") -> None:
    profile = apply_hardware(profile_with_gpu(CLOUD_MAIN_FULL_PROFILE), preset)
    run_pipeline(profile.steps, profile)


def run_cloud_main_no_finetune(preset: str = "server_24g") -> None:
    profile = apply_hardware(profile_with_gpu(CLOUD_MAIN_NO_FINETUNE_PROFILE), preset)
    run_pipeline(profile.steps, profile)
