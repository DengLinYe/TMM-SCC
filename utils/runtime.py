import os
import sys
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML

from .config import (
    DATA,
    AttackConfig,
    CONFIG_TEMPLATES,
    DATA_PATHS,
    ROOT,
    TMM_SCC,
    rel_path,
    subset_annotation_path,
)


def tmm_scc_env() -> dict:
    env = os.environ.copy()
    paths = [str(TMM_SCC), str(ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def write_runtime_config(
    task: str,
    subset: str,
    attack: AttackConfig,
    num_iters: Optional[int] = None,
) -> Path:
    yaml = YAML()
    yaml.preserve_quotes = True
    template = CONFIG_TEMPLATES[task]
    with open(template, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    ann_path = subset_annotation_path(task, subset)
    config["test_file"] = rel_path(ann_path)
    config["image_root"] = rel_path(DATA_PATHS["flickr_image_root"])
    config["bert_config"] = rel_path(TMM_SCC / "configs" / "config_bert.json")
    config["epsilon"] = attack.epsilon
    config["num_iters"] = num_iters if num_iters is not None else attack.num_iters
    config["alpha"] = attack.epsilon_per

    if task == "vlr":
        config["batch_size_test"] = attack.batch_size_vlr
        config["batch_size_train"] = attack.batch_size_vlr
    else:
        config["batch_size_test"] = attack.batch_size_ve
        config["batch_size_train"] = attack.batch_size_ve
        config["val_file"] = rel_path(DATA / "snli-ve" / f"ve_dev_{subset}.json") if (DATA / "snli-ve" / f"ve_dev_{subset}.json").exists() else config.get("val_file")
        config["train_file"] = [rel_path(DATA / "snli-ve" / f"ve_train_{subset}.json")] if (DATA / "snli-ve" / f"ve_train_{subset}.json").exists() else config.get("train_file")

    cache_dir = ROOT / "outputs" / "cache" / "configs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{num_iters}" if num_iters is not None else ""
    out_path = cache_dir / f"{task}_{subset}{suffix}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return out_path
