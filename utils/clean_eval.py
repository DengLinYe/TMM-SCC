import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    AttackConfig,
    CLIP_HF_ID,
    CONFIG_TEMPLATES,
    OUTPUTS,
    checkpoint_dir,
    flickr_image_root_for_task,
    is_hf_model,
    rel_path,
    resolve_checkpoint,
    resolve_clip_checkpoint,
    subset_annotation_path,
)
from .eval_cmd import build_eval_command
from .log import info, run_subprocess, skip
from .runtime import tmm_scc_env, write_runtime_config


def clean_metrics_path(task: str) -> Path:
    return checkpoint_dir(task) / "clean_metrics.json"


def _clean_eval_tmp_path(task: str, model: str, subset: str) -> Path:
    return OUTPUTS / ".cache" / "clean_eval" / f"{task}_{model}_{subset}.json"


def _clean_eval_work_dir(task: str, model: str, subset: str) -> Path:
    return OUTPUTS / ".cache" / "clean_eval" / f"{task}_{model}_{subset}"


def load_all_clean_metrics(task: str) -> Dict[str, Any]:
    path = clean_metrics_path(task)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def get_clean_record(task: str, subset: str, model: str) -> Optional[Dict[str, Any]]:
    return load_all_clean_metrics(task).get(subset, {}).get(model)


def clean_record_is_stale(task: str, model: str, metrics: Optional[Dict[str, Any]]) -> bool:
    if not metrics:
        return False
    if task == "vlr" and model == "blip":
        return float(metrics.get("r_mean", 0.0)) < 80.0
    return False


def save_clean_record(
    task: str, subset: str, model: str, metrics: Dict[str, Any]
) -> None:
    path = clean_metrics_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_all_clean_metrics(task)
    data.setdefault(subset, {})[model] = metrics
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_clean_record(task: str, subset: str, model: str) -> None:
    path = clean_metrics_path(task)
    if not path.is_file():
        return
    data = load_all_clean_metrics(task)
    if subset in data and model in data[subset]:
        del data[subset][model]
        if not data[subset]:
            del data[subset]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_blackbox_victim_clean_command(
    task: str,
    model: str,
    subset: str,
    attack: AttackConfig,
    dump_path: Path,
) -> list:
    ann = subset_annotation_path(task, subset)
    img_root = flickr_image_root_for_task(task)
    config_path = CONFIG_TEMPLATES[task]
    cmd = [
        sys.executable,
        "tmm_scc/EvalBlackBox.py",
        "--task",
        task,
        "--target",
        model,
        "--subset",
        subset,
        "--annotation",
        rel_path(ann),
        "--image-root",
        rel_path(img_root),
        "--config",
        rel_path(config_path),
        "--gpu",
        str(attack.gpu),
        "--clean-eval",
        "--dump-metrics",
        rel_path(dump_path),
    ]
    if is_hf_model(task, model):
        cmd.extend(["--checkpoint", resolve_clip_checkpoint()])
    else:
        checkpoint = resolve_checkpoint(task, model)
        cmd.extend(["--checkpoint", rel_path(checkpoint)])
    return cmd


def run_blackbox_victim_clean_eval(
    task: str,
    model: str,
    subset: str,
    attack: AttackConfig,
    dry_run: bool = False,
) -> int:
    dump_path = _clean_eval_tmp_path(task, model, subset)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_blackbox_victim_clean_command(
        task, model, subset, attack, dump_path
    )
    info(f"clean eval: {task}/{model} @ {subset}", module="clean_eval")
    if dry_run:
        from .log import cmdline

        cmdline(cmd)
        return 0

    if dump_path.exists():
        dump_path.unlink()

    rc = run_subprocess(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        env=tmm_scc_env(),
        module="clean_eval",
        label=f"clean {task}/{model}",
    )
    if rc != 0:
        return rc
    if not dump_path.is_file():
        skip(f"clean eval 未生成指标 {dump_path}", module="clean_eval")
        return 1

    metrics = json.loads(dump_path.read_text(encoding="utf-8"))
    dump_path.unlink(missing_ok=True)
    save_clean_record(task, subset, model, metrics)
    info(f"clean 指标已写入 {clean_metrics_path(task)}", module="clean_eval")
    return 0


def build_clean_eval_command(
    task: str,
    model: str,
    subset: str,
    attack: AttackConfig,
    dump_path: Path,
) -> list:
    config_path = write_runtime_config(task, subset, attack)
    checkpoint = resolve_checkpoint(task, model)
    out_dir = _clean_eval_work_dir(task, model, subset)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_eval_command(
        task,
        "tmm",
        config_path,
        checkpoint,
        out_dir,
        attack,
        subset,
        model,
    )
    adv_idx = cmd.index("--adv")
    cmd[adv_idx + 1] = "0"
    cmd.extend(["--clean_eval", "--dump_metrics", rel_path(dump_path)])
    if "--run_log" in cmd:
        idx = cmd.index("--run_log")
        cmd.pop(idx)
        cmd.pop(idx)
    return cmd


def run_clean_eval(
    task: str,
    model: str,
    subset: str,
    attack: AttackConfig,
    dry_run: bool = False,
) -> int:
    if task == "vlr" and model in ("tcl", "blip"):
        return run_blackbox_victim_clean_eval(
            task, model, subset, attack, dry_run=dry_run
        )
    if is_hf_model(task, model):
        return run_clip_clean_eval(subset, attack, dry_run=dry_run)

    checkpoint = resolve_checkpoint(task, model)
    if checkpoint is None or not checkpoint.exists():
        skip(f"clean eval: checkpoint 不存在 {checkpoint}", module="clean_eval")
        return 1

    dump_path = _clean_eval_tmp_path(task, model, subset)
    work_dir = _clean_eval_work_dir(task, model, subset)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_clean_eval_command(task, model, subset, attack, dump_path)
    info(f"clean eval: {task}/{model} @ {subset}", module="clean_eval")
    if dry_run:
        from .log import cmdline

        cmdline(cmd)
        return 0

    if dump_path.exists():
        dump_path.unlink()

    rc = run_subprocess(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        env=tmm_scc_env(),
        module="clean_eval",
        label=f"clean {task}/{model}",
    )
    if rc != 0:
        return rc
    if not dump_path.is_file():
        skip(f"clean eval 未生成指标文件 {dump_path}", module="clean_eval")
        return 1

    metrics = json.loads(dump_path.read_text(encoding="utf-8"))
    dump_path.unlink(missing_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    save_clean_record(task, subset, model, metrics)
    info(f"clean 指标已写入 {clean_metrics_path(task)}", module="clean_eval")
    return 0


def ensure_clean_metrics(
    task: str,
    model: str,
    subset: str,
    attack: AttackConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    existing = get_clean_record(task, subset, model)
    refresh = force or clean_record_is_stale(task, model, existing)
    if not refresh and existing:
        skip(
            f"clean 指标已存在: {task}/{model}/{subset}",
            module="clean_eval",
        )
        return 0
    if refresh:
        clear_clean_record(task, subset, model)
    return run_clean_eval(task, model, subset, attack, dry_run=dry_run)


def refresh_ve_clean_after_finetune(
    backbone: str,
    subset: str,
    attack: AttackConfig,
    dry_run: bool = False,
) -> int:
    return ensure_clean_metrics(
        "ve", backbone, subset, attack, force=True, dry_run=dry_run
    )


def build_clip_clean_command(
    subset: str,
    attack: AttackConfig,
    dump_path: Path,
) -> list:
    ann = subset_annotation_path("vlr", subset)
    img_root = flickr_image_root_for_task("vlr")
    return [
        sys.executable,
        "tmm_scc/EvalBlackBox.py",
        "--task",
        "vlr",
        "--target",
        "clip",
        "--subset",
        subset,
        "--annotation",
        rel_path(ann),
        "--image-root",
        rel_path(img_root),
        "--checkpoint",
        resolve_clip_checkpoint(),
        "--clean-eval",
        "--dump-metrics",
        rel_path(dump_path),
        "--gpu",
        str(attack.gpu),
    ]


def run_clip_clean_eval(
    subset: str,
    attack: AttackConfig,
    dry_run: bool = False,
) -> int:
    dump_path = _clean_eval_tmp_path("vlr", "clip", subset)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_clip_clean_command(subset, attack, dump_path)
    info(f"clean eval: vlr/clip @ {subset}", module="clean_eval")
    if dry_run:
        from .log import cmdline

        cmdline(cmd)
        return 0

    if dump_path.exists():
        dump_path.unlink()

    rc = run_subprocess(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        env=tmm_scc_env(),
        module="clean_eval",
        label="clean vlr/clip",
    )
    if rc != 0:
        return rc
    if not dump_path.is_file():
        skip(f"CLIP clean eval 未生成指标 {dump_path}", module="clean_eval")
        return 1

    metrics = json.loads(dump_path.read_text(encoding="utf-8"))
    dump_path.unlink(missing_ok=True)
    save_clean_record("vlr", subset, "clip", metrics)
    info(f"clean 指标已写入 {clean_metrics_path('vlr')}", module="clean_eval")
    return 0


def ensure_blackbox_victim_clean_metrics(
    tasks: List[str],
    targets: Dict[str, List[str]],
    subset: str,
    attack: AttackConfig,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    rc = 0
    seen = set()
    for task in tasks:
        for model in targets.get(task, []):
            key = (task, model, subset)
            if key in seen:
                continue
            seen.add(key)
            existing = get_clean_record(task, subset, model)
            refresh = force or clean_record_is_stale(task, model, existing)
            if refresh:
                clear_clean_record(task, subset, model)
            if is_hf_model(task, model):
                if not refresh and existing:
                    skip(
                        f"clean 指标已存在: {task}/{model}/{subset}",
                        module="clean_eval",
                    )
                    continue
                r = run_clip_clean_eval(subset, attack, dry_run=dry_run)
            elif task == "vlr" and model in ("tcl", "blip"):
                if not refresh and existing:
                    skip(
                        f"clean 指标已存在: {task}/{model}/{subset}",
                        module="clean_eval",
                    )
                    continue
                r = run_blackbox_victim_clean_eval(
                    task, model, subset, attack, dry_run=dry_run
                )
            else:
                r = ensure_clean_metrics(
                    task, model, subset, attack, force=force, dry_run=dry_run
                )
            if r != 0:
                rc = r
    return rc
