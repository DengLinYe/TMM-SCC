import argparse
import subprocess
import sys
import time
from pathlib import Path

from .config import (
    CHECKPOINT_MAP,
    METHOD_LABELS,
    AttackConfig,
    ExperimentMatrix,
    output_dir,
    rel_path,
    resolve_checkpoint,
)
from .runtime import tmm_scc_env, write_runtime_config

EVAL_SCRIPTS = {
    "vlr": "EvalTransferAttack.py",
    "ve": "EvalVEAttack.py",
}


def build_eval_command(
    task: str,
    method: str,
    config_path: Path,
    checkpoint: Path,
    out_dir: Path,
    attack: AttackConfig,
) -> list:
    script = Path("tmm_scc") / EVAL_SCRIPTS[task]
    cmd = [
        sys.executable,
        str(script),
        "--adv",
        "1",
        "--gpu",
        str(attack.gpu),
        "--checkpoint",
        rel_path(checkpoint),
        "--config",
        rel_path(config_path),
        "--dataset",
        "flickr" if task == "vlr" else "snli-ve",
        "--text_method",
        method,
        "--sim_threshold",
        str(attack.sim_threshold),
        "--epsilon",
        str(attack.epsilon),
        "--epsilon_per",
        str(attack.epsilon_per),
        "--att_mask",
        str(attack.att_mask),
        "--kernel_size",
        str(attack.kernel_size),
        "--momentum",
        str(attack.momentum),
        "--intervals",
        str(attack.num_steps),
        "--output_dir",
        rel_path(out_dir),
        "--save_dir",
        rel_path(out_dir) + "/",
        "--log_name",
        "metrics.jsonl",
        "--config_name",
        "run_config.yaml",
    ]
    if attack.use_cls:
        cmd.append("--cls")
    return cmd


def run_single_attack(
    task: str,
    model: str,
    method: str,
    subset: str,
    attack: AttackConfig,
    num_iters: int = None,
    dry_run: bool = False,
) -> int:
    if method in ("coattack", "sga"):
        from baselines import run_baseline_attack

        return run_baseline_attack(
            task=task,
            method=method,
            subset=subset,
            attack=attack,
            num_iters=num_iters,
            dry_run=dry_run,
        )

    checkpoint = resolve_checkpoint(task, model)
    if checkpoint is None or not checkpoint.exists():
        if dry_run:
            checkpoint = CHECKPOINT_MAP.get(task, {}).get(model) or Path("checkpoints/missing.pth")
        else:
            print(f"[!] 跳过 {task}/{model}/{method}: checkpoint 不存在 {checkpoint}")
            return 1

    config_path = write_runtime_config(task, subset, attack, num_iters=num_iters)
    out_dir = output_dir(task, model, method, subset, num_iters=num_iters)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_eval_command(task, method, config_path, checkpoint, out_dir, attack)
    label = METHOD_LABELS.get(method, method)
    print(f"\n{'=' * 50}")
    print(f"[▶] 白盒攻击: task={task}, model={model}, method={label}, subset={subset}")
    print(f"{'=' * 50}")
    if dry_run:
        print(" ".join(cmd))
        return 0

    result = subprocess.run(cmd, env=tmm_scc_env(), cwd=str(Path(__file__).resolve().parent.parent))
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="运行白盒对抗攻击（TMM / TMM-SCC）")
    parser.add_argument("--task", choices=["vlr", "ve", "all"], default="all")
    parser.add_argument("--method", choices=["tmm", "scc", "all"], default="all")
    parser.add_argument("--subset", default="main_1k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--cooldown", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    attack = AttackConfig(gpu=args.gpu)
    if args.num_iters is not None:
        attack.num_iters = args.num_iters

    tasks = ["vlr", "ve"] if args.task == "all" else [args.task]
    methods = ["tmm", "scc"] if args.method == "all" else [args.method]
    matrix = ExperimentMatrix(tasks=tasks, methods=methods, subset=args.subset)

    failed = 0
    jobs = matrix.attack_jobs()
    for i, job in enumerate(jobs):
        rc = run_single_attack(
            task=job["task"],
            model=job["model"],
            method=job["method"],
            subset=job["subset"],
            attack=attack,
            num_iters=args.num_iters,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failed += 1
        elif not args.dry_run and i < len(jobs) - 1:
            print(f"[⏳] 休眠 {args.cooldown}s ...")
            time.sleep(args.cooldown)

    if failed:
        print(f"\n[!] 完成，{failed} 个任务失败")
        sys.exit(1)
    print("\n[+] 所有攻击任务完成")


if __name__ == "__main__":
    main()
