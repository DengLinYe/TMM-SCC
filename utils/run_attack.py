import argparse
import sys
import time
from pathlib import Path

from .clean_eval import ensure_clean_metrics
from .config import (
    CHECKPOINT_MAP,
    METHOD_LABELS,
    AttackConfig,
    ExperimentMatrix,
    RunProfile,
    attack_config_for,
    output_dir,
    rel_path,
    resolve_checkpoint,
    resolve_whitebox_methods,
)
from .eval_cmd import build_eval_command
from .log import cmdline, finish_fail, finish_ok, header, run_subprocess, skip, wait
from .output import reset_output_dir
from .runtime import tmm_scc_env, write_runtime_config


def run_single_attack(
    task: str,
    model: str,
    method: str,
    subset: str,
    attack: AttackConfig,
    num_iters: int = None,
    ablation: bool = False,
    dry_run: bool = False,
) -> int:
    if not resolve_whitebox_methods(task, method):
        skip(f"{METHOD_LABELS.get(method, method)} 不支持任务 {task}", module="attack")
        return 0

    if method in ("coattack", "sga"):
        from baselines import run_baseline_attack

        rc = ensure_clean_metrics(
            task, model, subset, attack, dry_run=dry_run
        )
        if rc != 0:
            return rc
        return run_baseline_attack(
            task=task,
            method=method,
            subset=subset,
            attack=attack,
            num_iters=num_iters,
            dry_run=dry_run,
        )

    rc = ensure_clean_metrics(task, model, subset, attack, dry_run=dry_run)
    if rc != 0:
        return rc

    checkpoint = resolve_checkpoint(task, model)
    if checkpoint is None or not checkpoint.exists():
        if dry_run:
            checkpoint = CHECKPOINT_MAP.get(task, {}).get(model) or Path("checkpoints/missing.pth")
        else:
            skip(f"{task}/{model}/{method}: checkpoint 不存在 {checkpoint}", module="attack")
            return 1

    config_path = write_runtime_config(
        task, subset, attack, num_iters=num_iters, dry_run=dry_run
    )
    iter_suffix = num_iters if ablation else None
    out_dir = output_dir(task, model, method, subset, num_iters=iter_suffix)
    if dry_run:
        cmd = build_eval_command(
            task, method, config_path, checkpoint, out_dir, attack, subset, model
        )
        label = METHOD_LABELS.get(method, method)
        header(f"白盒攻击: task={task}, model={model}, method={label}, subset={subset}")
        cmdline(cmd)
        return 0

    reset_output_dir(out_dir)

    cmd = build_eval_command(
        task, method, config_path, checkpoint, out_dir, attack, subset, model
    )
    label = METHOD_LABELS.get(method, method)
    header(f"白盒攻击: task={task}, model={model}, method={label}, subset={subset}")

    return run_subprocess(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        env=tmm_scc_env(),
        module="attack",
        label=f"白盒攻击 {task}/{method}",
    )


def main(argv=None, profile: RunProfile = None):
    parser = argparse.ArgumentParser(description="运行白盒对抗攻击（TMM / SCC / Co-Attack / SGA）")
    parser.add_argument("--task", choices=["vlr", "ve", "all"], default="all")
    parser.add_argument("--method", choices=["tmm", "scc", "coattack", "sga", "all"], default="all")
    parser.add_argument("--subset", default="main_1k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--cooldown", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if profile is not None:
        attack = attack_config_for(profile, gpu=profile.gpu)
        task = profile.attack_task
        method = profile.attack_method
        subset = profile.subset
        cooldown = profile.cooldown
        dry_run = profile.dry_run or args.dry_run
        num_iters_override = None
    else:
        attack = attack_config_for(gpu=args.gpu, num_iters=args.num_iters)
        task = args.task
        method = args.method
        subset = args.subset
        cooldown = args.cooldown
        dry_run = args.dry_run
        num_iters_override = args.num_iters

    tasks = ["vlr", "ve"] if task == "all" else [task]
    matrix = ExperimentMatrix(tasks=tasks, method=method, subset=subset)

    jobs = matrix.attack_jobs()
    if not jobs and not dry_run:
        skip(f"无匹配攻击任务: task={task}, method={method}", module="attack")
        return 1
    if not jobs:
        finish_ok("无匹配攻击任务 (dry-run)", module="attack")
        return 0

    failed = 0
    for i, job in enumerate(jobs):
        rc = run_single_attack(
            task=job["task"],
            model=job["model"],
            method=job["method"],
            subset=job["subset"],
            attack=attack,
            num_iters=num_iters_override,
            dry_run=dry_run,
        )
        if rc != 0:
            failed += 1
        elif not dry_run and i < len(jobs) - 1:
            wait(f"休眠 {cooldown}s ...", module="attack")
            time.sleep(cooldown)

    if failed:
        finish_fail(failed, module="attack")
    finish_ok("所有攻击任务完成", module="attack")


if __name__ == "__main__":
    main()
