import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

from ruamel.yaml import YAML

from utils.config import (
    ROOT,
    TMM_SCC,
    AttackConfig,
    DATA,
    OUTPUTS,
    SURROGATE_MODEL,
    apply_runtime_env,
    flickr_image_root_for_task,
    rel_path,
    run_log_path,
    subset_annotation_path,
)
from utils.log import require_file
from utils.runtime import _vlr_dataset_sizes

BASELINE_ROOTS = {
    "coattack": ROOT / "baselines" / "co_attack",
    "sga": ROOT / "baselines" / "sga",
}

BASELINE_TEMPLATES = {
    "coattack": {
        "vlr": BASELINE_ROOTS["coattack"] / "configs" / "Retrieval_flickr.yaml",
        "ve": BASELINE_ROOTS["coattack"] / "configs" / "VE.yaml",
    },
    "sga": {
        "vlr": BASELINE_ROOTS["sga"] / "configs" / "Retrieval_flickr.yaml",
    },
}


def baseline_env(baseline_dir: Path) -> dict:
    env = os.environ.copy()
    env = apply_runtime_env(env)
    paths = [
        str(baseline_dir.resolve()),
        str(TMM_SCC.resolve()),
        str(ROOT.resolve()),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def path_from_baseline(target: Path, baseline_dir: Path) -> str:
    return os.path.relpath(target.resolve(), baseline_dir.resolve()).replace("\\", "/")


def write_baseline_runtime_config(
    method: str,
    task: str,
    subset: str,
    attack: AttackConfig,
    num_iters: Optional[int] = None,
    dry_run: bool = False,
) -> Path:
    yaml = YAML()
    yaml.preserve_quotes = True
    template = BASELINE_TEMPLATES[method][task]
    baseline_dir = BASELINE_ROOTS[method]
    require_file(template, f"{method} {task} 配置模板", module="baseline")
    with open(template, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    ann_path = subset_annotation_path(task, subset)
    if not dry_run:
        require_file(ann_path, f"{task} 子集标注 ({subset})", module="baseline")

    config["test_file"] = path_from_baseline(ann_path, baseline_dir)
    config["image_root"] = path_from_baseline(flickr_image_root_for_task(task), baseline_dir).rstrip("/") + "/"
    config["bert_config"] = path_from_baseline(baseline_dir / "configs" / "config_bert.json", baseline_dir)
    config["epsilon"] = attack.epsilon
    config["num_iters"] = num_iters if num_iters is not None else attack.num_iters
    if "alpha" in config:
        config["alpha"] = attack.epsilon_per

    if task == "vlr":
        config["batch_size_test"] = attack.batch_size_vlr
        if ann_path.is_file():
            num_images, num_texts = _vlr_dataset_sizes(ann_path)
            k = int(config.get("k_test", 64))
            config["k_test"] = max(1, min(k, num_images, num_texts))
    else:
        config["batch_size_test"] = attack.batch_size_ve
        dev = DATA / "snli-ve" / f"ve_dev_{subset}.json"
        if dev.is_file():
            config["val_file"] = path_from_baseline(dev, baseline_dir)

    cache = OUTPUTS / "cache" / "configs"
    cache.mkdir(parents=True, exist_ok=True)
    suffix = f"_{num_iters}" if num_iters is not None else ""
    out_path = cache / f"{method}_{task}_{subset}{suffix}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return out_path


def _ve_accuracy_percent(raw) -> float:
    """Co-Attack VEEval 写入 0~1 比例；TMM/clean 指标为 0~100 百分制。"""
    acc = float(raw)
    if acc <= 1.0:
        acc *= 100.0
    return acc


def parse_coattack_log(out_dir: Path, task: str) -> Dict[str, float]:
    log_path = out_dir / "log.txt"
    if not log_path.is_file():
        raise FileNotFoundError(f"Co-Attack 未生成日志 {log_path}")
    line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    stats = json.loads(line)
    if task == "vlr":
        return {
            "txt_r1": float(stats["test_txt_r1"]),
            "txt_r5": float(stats["test_txt_r5"]),
            "txt_r10": float(stats["test_txt_r10"]),
            "img_r1": float(stats["test_img_r1"]),
            "img_r5": float(stats["test_img_r5"]),
            "img_r10": float(stats["test_img_r10"]),
            "txt_r_mean": float(stats["test_txt_r_mean"]),
            "img_r_mean": float(stats["test_img_r_mean"]),
            "r_mean": float(stats["test_r_mean"]),
            "avg_sim": 0.0,
        }
    return {"accuracy": _ve_accuracy_percent(stats["test_acc"]), "avg_sim": 0.0}


def parse_sga_metric(value: str) -> float:
    match = re.search(r"\(([\d.]+)\)", str(value))
    if match:
        return float(match.group(1))
    return float(value)


def log_whitebox_baseline(
    method: str,
    task: str,
    model: str,
    subset: str,
    attack: AttackConfig,
    raw_result: dict,
    checkpoint: Path,
    output_dir: Path,
) -> None:
    import tmm_scc.run_log as run_log
    from tmm_scc.utils import append_run_log, load_clean_raw

    experiment = f"whitebox_{task}_{model}_{method}"
    cfg = {
        "task": task,
        "subset": subset,
        "model": model,
        "method": method,
        "checkpoint": rel_path(checkpoint),
        "output_dir": rel_path(output_dir),
        "epsilon": attack.epsilon,
        "num_iters": attack.num_iters,
    }
    clean = load_clean_raw(task, subset, model)
    if task == "vlr":
        results = run_log.vlr_results(raw_result, clean=clean)
    else:
        clean_acc = clean.get("accuracy") if clean else None
        results = run_log.ve_results(
            raw_result["accuracy"],
            raw_result.get("avg_sim", 0.0),
            clean_accuracy=clean_acc,
        )
    append_run_log(
        str(run_log_path()),
        run_log.build_experiment_record(experiment, cfg, results),
    )
