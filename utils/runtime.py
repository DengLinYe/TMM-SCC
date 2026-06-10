"""攻击/评测子进程环境与运行时 YAML（由模板 + AttackConfig + 子集路径生成）。"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML

from .config import (
    DATA,
    AttackConfig,
    CONFIG_TEMPLATES,
    OUTPUTS,
    ROOT,
    TMM_SCC,
    apply_runtime_env,
    flickr_image_root_for_task,
    rel_path,
    subset_annotation_path,
)

from .log import info, require_file


def _vlr_dataset_sizes(ann_path: Path) -> tuple:
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_ids = set()
    num_text = 0
    for entry in data:
        img_id = entry.get("image") or entry.get("image_id")
        img_ids.add(img_id)
        captions = entry.get("caption")
        if captions is None:
            num_text += 1
        elif isinstance(captions, list):
            num_text += len(captions)
        else:
            num_text += 1
    return len(img_ids), num_text


def tmm_scc_env() -> dict:
    env = os.environ.copy()
    env = apply_runtime_env(env)
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
    dry_run: bool = False,
) -> Path:
    yaml = YAML()
    yaml.preserve_quotes = True
    template = CONFIG_TEMPLATES[task]
    require_file(template, f"{task} 配置模板", module="runtime")
    with open(template, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    ann_path = subset_annotation_path(task, subset)
    if dry_run:
        if not ann_path.is_file():
            info(
                f"dry-run: 子集标注未生成，使用预期路径 {rel_path(ann_path)}",
                module="runtime",
            )
    else:
        require_file(ann_path, f"{task} 子集标注 ({subset})", module="runtime")
    config["test_file"] = rel_path(ann_path)
    config["image_root"] = rel_path(flickr_image_root_for_task(task)) + "/"
    config["bert_config"] = rel_path(TMM_SCC / "configs" / "config_bert.json")
    config["epsilon"] = attack.epsilon
    config["num_iters"] = num_iters if num_iters is not None else attack.num_iters
    config["alpha"] = attack.epsilon_per

    if task == "vlr":
        config["batch_size_test"] = attack.batch_size_vlr
        config["batch_size_train"] = attack.batch_size_vlr
        if ann_path.is_file():
            num_images, num_texts = _vlr_dataset_sizes(ann_path)
            k = int(config.get("k_test", 64))
            # ALBEF 检索 rerank：k 不能超过当前子集的图像/文本数（任意规模均适用）
            config["k_test"] = max(1, min(k, num_images, num_texts))
    else:
        config["batch_size_test"] = attack.batch_size_ve
        config["batch_size_train"] = attack.batch_size_ve
        config["val_file"] = rel_path(DATA / "snli-ve" / f"ve_dev_{subset}.json") if (DATA / "snli-ve" / f"ve_dev_{subset}.json").exists() else config.get("val_file")
        config["train_file"] = [rel_path(DATA / "snli-ve" / f"ve_train_{subset}.json")] if (DATA / "snli-ve" / f"ve_train_{subset}.json").exists() else config.get("train_file")

    cache_dir = OUTPUTS / "cache" / "configs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{num_iters}" if num_iters is not None else ""
    out_path = cache_dir / f"{task}_{subset}{suffix}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return out_path
