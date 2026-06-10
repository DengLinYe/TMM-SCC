import argparse
import sys
from pathlib import Path

from .clean_eval import ensure_blackbox_victim_clean_metrics
from .config import (
    BLACKBOX_TARGETS,
    METHOD_LABELS,
    RunProfile,
    attack_config_for,
    blackbox_log_path,
    flickr_image_root_for_task,
    is_hf_model,
    output_dir,
    rel_path,
    resolve_checkpoint,
    resolve_clip_checkpoint,
    resolve_whitebox_methods,
    subset_annotation_path,
)
from .log import cmdline, finish_fail, finish_ok, run_subprocess, skip, step
from .runtime import tmm_scc_env

DEFAULT_TARGETS = BLACKBOX_TARGETS


def manifest_path(task: str, model: str, method: str, subset: str) -> Path:
    out = output_dir(task, model, method, subset)
    name = "ve_adv_manifest.json" if task == "ve" else "vlr_adv_manifest.json"
    return out / name


def adv_samples_dir(task: str, model: str, method: str, subset: str) -> Path:
    out = output_dir(task, model, method, subset)
    sub = "adv_samples_ve" if task == "ve" else "adv_samples_retrieval"
    return out / sub


def build_blackbox_jobs(tasks, method_sel, targets, subset):
    jobs = []
    for task in tasks:
        for method in resolve_whitebox_methods(task, method_sel):
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
        return resolve_clip_checkpoint()
    ckpt = resolve_checkpoint(task, target)
    if ckpt is None:
        return ""
    return rel_path(ckpt)


def run_blackbox_eval(job: dict, dry_run: bool = False, gpu: int = 0) -> int:
    manifest = manifest_path(job["task"], job["surrogate"], job["method"], job["subset"])
    adv_dir = adv_samples_dir(job["task"], job["surrogate"], job["method"], job["subset"])

    if not manifest.exists():
        skip(f"manifest 不存在，跳过 {manifest}", module="blackbox")
        return 0

    ckpt = checkpoint_arg(job["task"], job["target"])
    if not is_hf_model(job["task"], job["target"]):
        path = resolve_checkpoint(job["task"], job["target"])
        if path is None or not path.exists():
            skip(f"checkpoint 不存在，跳过 {path}", module="blackbox")
            return 0

    ann = subset_annotation_path(job["task"], job["subset"])
    if not ann.is_file():
        skip(f"子集标注不存在 {ann}", module="blackbox")
        return 0

    img_root = flickr_image_root_for_task(job["task"])

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
        "--annotation",
        rel_path(ann),
        "--image-root",
        rel_path(img_root),
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
    step(f"黑盒: {job['task']} | {label} -> {job['target']}", module="blackbox")
    if dry_run:
        cmdline(cmd)
        return 0

    return run_subprocess(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        env=tmm_scc_env(),
        module="blackbox",
        label=f"黑盒 {job['task']} {job['method']}->{job['target']}",
    )


def main(argv=None, profile: RunProfile = None):
    parser = argparse.ArgumentParser(description="黑盒迁移攻击评测")
    parser.add_argument("--task", choices=["vlr", "ve", "all"], default="all")
    parser.add_argument("--method", choices=["tmm", "scc", "coattack", "sga", "all"], default="all")
    parser.add_argument("--target", nargs="+", default=None)
    parser.add_argument("--subset", default="main_1k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    gpu = profile.gpu if profile is not None else args.gpu
    dry_run = (profile.dry_run if profile is not None else False) or args.dry_run
    subset = profile.subset if profile is not None else args.subset
    task_sel = profile.blackbox_task if profile is not None else args.task
    method_sel = profile.blackbox_method if profile is not None else args.method

    tasks = ["vlr", "ve"] if task_sel == "all" else [task_sel]
    targets = {k: list(v) for k, v in DEFAULT_TARGETS.items()}
    if profile and profile.blackbox_targets:
        for k, v in profile.blackbox_targets.items():
            targets[k] = list(v)
    if args.target:
        for t in tasks:
            targets[t] = args.target

    attack = attack_config_for(profile, gpu=gpu) if profile else attack_config_for(gpu=gpu)

    rc = ensure_blackbox_victim_clean_metrics(
        tasks,
        targets,
        subset,
        attack,
        dry_run=dry_run,
        force=bool(profile and profile.force_finetune),
    )
    if rc != 0 and not dry_run:
        finish_fail(1, label="受害模型 clean 基准", module="blackbox")
        return rc

    blackbox_log_path().parent.mkdir(parents=True, exist_ok=True)
    jobs = build_blackbox_jobs(tasks, method_sel, targets, subset)
    for job in jobs:
        job["gpu"] = gpu

    failed = 0
    for job in jobs:
        if run_blackbox_eval(job, dry_run=dry_run, gpu=gpu) != 0:
            failed += 1

    if failed:
        finish_fail(failed, label="黑盒评测", module="blackbox")
    finish_ok("黑盒评测全部完成", module="blackbox")
