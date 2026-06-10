import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import OUTPUTS, run_log_path

TMM_SCC = Path(__file__).resolve().parent.parent / "tmm_scc"
if str(TMM_SCC) not in sys.path:
    sys.path.insert(0, str(TMM_SCC))

import run_log

RUN_LOG_NAME = run_log.RUN_LOG_NAME
PERSISTENT_OUTPUTS = run_log.PERSISTENT_OUTPUTS

__all__ = [
    "RUN_LOG_NAME",
    "PERSISTENT_OUTPUTS",
    "append_run_log",
    "build_experiment_record",
    "clean_ephemeral_outputs",
    "clean_for_cloud_main",
    "reset_output_dir",
]


def append_run_log(record: Dict[str, Any], path: Optional[Path] = None) -> None:
    run_log.append_run_log(record, path or run_log_path())


def build_experiment_record(
    experiment: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    return run_log.build_experiment_record(experiment, config, results)


def clean_ephemeral_outputs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for child in OUTPUTS.iterdir():
        if child.name in PERSISTENT_OUTPUTS or child.name == "run.log":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def clean_for_cloud_main(
    clear_subsets=("mini_100", "main_1k"),
    preserve_subsets=("ablation_200",),
) -> None:
    """正式集跑前：删测试/正式攻击产物，保留 run.json 与 ablation_200 消融目录。"""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    whitebox = OUTPUTS / "whitebox"
    if whitebox.is_dir():
        for path in sorted(whitebox.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if not path.is_dir():
                continue
            if path.name in preserve_subsets:
                continue
            if path.name in clear_subsets:
                shutil.rmtree(path, ignore_errors=True)
                continue
            if path.name.startswith("iter_") and path.parent.name in clear_subsets:
                shutil.rmtree(path, ignore_errors=True)
    for child in OUTPUTS.iterdir():
        if child.name in PERSISTENT_OUTPUTS or child.name == "whitebox":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        elif child.name != "run.log":
            child.unlink(missing_ok=True)


def reset_output_dir(path: Path) -> Path:
    from .log import info

    if path.exists():
        try:
            label = path.relative_to(OUTPUTS).as_posix()
        except ValueError:
            label = str(path)
        info(f"清理输出目录 outputs/{label}", module="output")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
