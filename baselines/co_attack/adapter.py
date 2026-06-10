import json
import sys
from pathlib import Path

from utils.config import SURROGATE_MODEL, output_dir, resolve_checkpoint
from utils.log import cmdline, header, run_subprocess, skip
from utils.output import reset_output_dir

from baselines.common import (
    BASELINE_ROOTS,
    baseline_env,
    log_whitebox_baseline,
    parse_coattack_log,
    path_from_baseline,
    write_baseline_runtime_config,
)

CO_ATTACK_ROOT = BASELINE_ROOTS["coattack"]
SCRIPTS = {"vlr": "RetrievalEval.py", "ve": "VEEval.py"}


class CoAttackAdapter:
    def run(self, task, subset, attack, num_iters=None, dry_run=False) -> int:
        if task not in SCRIPTS:
            skip(f"Co-Attack 不支持任务 {task}", module="baseline")
            return 1

        model = SURROGATE_MODEL.get(task, "albef")
        checkpoint = resolve_checkpoint(task, model)
        if checkpoint is None or not checkpoint.exists():
            if not dry_run:
                skip(f"Co-Attack: checkpoint 不存在 {checkpoint}", module="baseline")
                return 1
            checkpoint = Path("checkpoints/missing.pth")

        config_path = write_baseline_runtime_config(
            "coattack", task, subset, attack, num_iters=num_iters, dry_run=dry_run
        )
        out_dir = output_dir(task, model, "coattack", subset)

        cmd = [
            sys.executable,
            SCRIPTS[task],
            "--adv",
            str(attack.coattack_adv),
            "--cls",
            "--gpu",
            str(attack.gpu),
            "--checkpoint",
            path_from_baseline(checkpoint, CO_ATTACK_ROOT),
            "--config",
            path_from_baseline(config_path, CO_ATTACK_ROOT),
            "--output_dir",
            path_from_baseline(out_dir, CO_ATTACK_ROOT),
            "--save_dir",
            path_from_baseline(out_dir, CO_ATTACK_ROOT) + "/",
            "--alpha",
            str(attack.epsilon_per),
        ]

        header(f"Co-Attack: task={task}, model={model}, subset={subset}")
        if dry_run:
            cmdline(cmd)
            return 0

        reset_output_dir(out_dir)
        rc = run_subprocess(
            cmd,
            cwd=CO_ATTACK_ROOT,
            env=baseline_env(CO_ATTACK_ROOT),
            module="baseline",
            label=f"Co-Attack {task}",
        )
        if rc != 0:
            return rc

        try:
            raw = parse_coattack_log(out_dir, task)
            log_whitebox_baseline(
                "coattack", task, model, subset, attack, raw, checkpoint, out_dir
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            skip(f"Co-Attack 结果解析失败: {exc}", module="baseline")
            return 1
        return 0
