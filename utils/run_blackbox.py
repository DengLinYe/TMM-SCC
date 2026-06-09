import argparse
import subprocess
import sys
from pathlib import Path

from .config import (
    CLIP_HF_ID,
    HF_VLR_MODELS,
    METHOD_LABELS,
    blackbox_log_path,
    is_hf_model,
    output_dir,
    rel_path,
    resolve_checkpoint,
)
from .runtime import tmm_scc_env

DEFAULT_TARGETS = {
    "vlr": ["tcl", "clip"],
    "ve": ["tcl"],
}


def manifest_path(task: str, model: str, method: str, subset: str) -> Path:
    out = output_dir(task, model, method, subset)
    name = "ve_adv_manifest.json" if task == "ve" else "vlr_adv_manifest.json"
    return out / name


def adv_samples_dir(task: str, model: str, method: str, subset: str) -> Path:
    out = output_dir(task, model, method, subset)
    sub = "adv_samples_ve" if task == "ve" else "adv_samples_retrieval"
    return out / sub


def build_blackbox_jobs(tasks, methods, targets, subset):
    jobs = []
    for task in tasks:
        for method in methods:
            for target in targets.get(task, []):
                jobs.append(
                    {
                        "task": task,
                        "surrogate": "albef",
                        "method": method,
                        "target": target,
                        "subset": subset,
                    }
                )
    return jobs


def checkpoint_arg(task: str, target: str) -> str:
    if is_hf_model(task, target):
        return f"hf:{HF_VLR_MODELS[target]}"
    ckpt = resolve_checkpoint(task, target)
    if ckpt is None:
        return ""
    return rel_path(ckpt)


def run_blackbox_eval(job: dict, dry_run: bool = False, gpu: int = 0) -> int:
    manifest = manifest_path(job["task"], job["surrogate"], job["method"], job["subset"])
    adv_dir = adv_samples_dir(job["task"], job["surrogate"], job["method"], job["subset"])

    if not manifest.exists():
        print(f"[!] 跳过: manifest 不存在 {manifest}")
        return 1

    ckpt = checkpoint_arg(job["task"], job["target"])
    if not is_hf_model(job["task"], job["target"]):
        path = resolve_checkpoint(job["task"], job["target"])
        if path is None or not path.exists():
            print(f"[!] 跳过: checkpoint 不存在 {path}")
            return 1

    cmd = [
        sys.executable,
        "tmm_scc/EvalBlackBox.py",
        "--task",
        job["task"],
        "--target",
        job["target"],
        "--method",
        job["method"],
        "--manifest",
        rel_path(manifest),
        "--adv-image-root",
        rel_path(adv_dir),
        "--checkpoint",
        ckpt,
        "--subset",
        job["subset"],
        "--log",
        rel_path(blackbox_log_path()),
        "--gpu",
        str(job.get("gpu", 0)),
    ]

    label = METHOD_LABELS.get(job["method"], job["method"])
    print(f"\n[▶] 黑盒: {job['task']} | {label} -> {job['target']}")
    if dry_run:
        print(" ".join(cmd))
        return 0

    return subprocess.run(
        cmd, env=tmm_scc_env(), cwd=str(Path(__file__).resolve().parent.parent)
    ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="黑盒迁移攻击评测")
    parser.add_argument("--task", choices=["vlr", "ve", "all"], default="all")
    parser.add_argument("--method", choices=["tmm", "scc", "all"], default="all")
    parser.add_argument("--target", nargs="+", default=None)
    parser.add_argument("--subset", default="main_1k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    tasks = ["vlr", "ve"] if args.task == "all" else [args.task]
    methods = ["tmm", "scc"] if args.method == "all" else [args.method]
    targets = DEFAULT_TARGETS.copy()
    if args.target:
        for t in tasks:
            targets[t] = args.target

    blackbox_log_path().parent.mkdir(parents=True, exist_ok=True)
    jobs = build_blackbox_jobs(tasks, methods, targets, args.subset)
    for job in jobs:
        job["gpu"] = args.gpu

    failed = 0
    for job in jobs:
        if run_blackbox_eval(job, dry_run=args.dry_run, gpu=args.gpu) != 0:
            failed += 1

    if failed:
        print(f"\n[!] 黑盒评测完成，{failed} 个任务失败")
        sys.exit(1)
    print("\n[+] 黑盒评测全部完成")
