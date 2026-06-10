"""TMM/SCC 子进程命令拼装（内部用，用户请走 python main.py 控制台）。"""

import sys
from pathlib import Path

from .config import AttackConfig, rel_path, run_log_path

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
    subset: str = "",
    model: str = "albef",
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
        "metrics.json",
        "--config_name",
        "run_config.yaml",
        "--run_log",
        rel_path(run_log_path()),
    ]
    if subset:
        cmd.extend(["--subset", subset])
    if model:
        cmd.extend(["--model", model])
    if attack.use_cls:
        cmd.append("--cls")
    return cmd
