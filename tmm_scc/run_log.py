"""实验结果汇总：追加写入 outputs/run.json（与 utils/log.py 终端输出无关）。"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

RUN_LOG_NAME = "run.json"
PERSISTENT_OUTPUTS = {RUN_LOG_NAME}


def _load_records(log_path: Path) -> List[dict]:
    if not log_path.exists() or log_path.stat().st_size == 0:
        legacy = log_path.with_name("run.log")
        if legacy.exists() and legacy.stat().st_size > 0:
            log_path = legacy
        else:
            return []
    text = log_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def append_run_log(record: Dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if "timestamp" not in record:
        record = {
            **record,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    records = _load_records(log_path)
    records.append(record)
    log_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _round4(value: float) -> float:
    return round(float(value), 4)


def vlr_accuracy(raw: Dict[str, float]) -> float:
    return (raw["txt_r1"] + raw["img_r1"]) / 2.0


def vlr_results(
    raw: Dict[str, float], clean: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    adv_acc = vlr_accuracy(raw)
    clean_acc = vlr_accuracy(clean) if clean else None
    if clean_acc is not None and clean_acc > 0:
        asr = 100.0 * (clean_acc - adv_acc) / clean_acc
    else:
        asr = None
    out = {
        "accuracy": _round4(adv_acc),
        "txt_r1": _round4(raw["txt_r1"]),
        "txt_r5": _round4(raw["txt_r5"]),
        "txt_r10": _round4(raw["txt_r10"]),
        "img_r1": _round4(raw["img_r1"]),
        "img_r5": _round4(raw["img_r5"]),
        "img_r10": _round4(raw["img_r10"]),
        "txt_r_mean": _round4(raw["txt_r_mean"]),
        "img_r_mean": _round4(raw["img_r_mean"]),
        "r_mean": _round4(raw["r_mean"]),
        "avg_sim": _round4(raw.get("avg_sim", 0.0)),
    }
    if clean_acc is not None:
        out["clean_accuracy"] = _round4(clean_acc)
        out["asr"] = _round4(max(0.0, asr))
    return out


def ve_results(
    accuracy: float,
    avg_sim: float = 0.0,
    clean_accuracy: Optional[float] = None,
) -> Dict[str, float]:
    accuracy = float(accuracy)
    out = {
        "accuracy": _round4(accuracy),
        "avg_sim": _round4(avg_sim),
    }
    if clean_accuracy is not None and clean_accuracy > 0:
        out["clean_accuracy"] = _round4(clean_accuracy)
        out["asr"] = _round4(
            max(0.0, 100.0 * (float(clean_accuracy) - accuracy) / float(clean_accuracy))
        )
    return out


def subset_dataset_counts(
    subset: str,
    task: str,
    *,
    num_images: Optional[int] = None,
    num_texts: Optional[int] = None,
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    counts: Dict[str, Any] = {"subset": subset}
    if task == "vlr":
        if num_images is not None:
            counts["num_images"] = num_images
        if num_texts is not None:
            counts["num_texts"] = num_texts
    elif num_samples is not None:
        counts["num_samples"] = num_samples
    return counts


def attack_config_block(args, config: dict, task: str) -> Dict[str, Any]:
    block = {
        "task": task,
        "dataset": args.dataset,
        "subset": getattr(args, "subset", ""),
        "model": getattr(args, "model", "albef"),
        "method": getattr(args, "text_method", getattr(args, "method", "")),
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
        "adv": args.adv,
        "cls": getattr(args, "cls", False),
        "epsilon": config.get("epsilon", getattr(args, "epsilon", None)),
        "num_iters": config.get("num_iters", getattr(args, "num_iters", None)),
        "alpha": config.get("alpha", getattr(args, "epsilon_per", None)),
        "batch_size": config.get("batch_size_test"),
        "sim_threshold": getattr(args, "sim_threshold", None),
        "intervals": getattr(args, "intervals", None),
        "kernel_size": getattr(args, "kernel_size", None),
        "momentum": getattr(args, "momentum", None),
        "mode": getattr(args, "mode", None),
    }
    return {k: v for k, v in block.items() if v is not None and v != ""}


def build_experiment_record(
    experiment: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment": experiment,
        "config": config,
        "results": results,
    }
