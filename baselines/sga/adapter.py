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
    path_from_baseline,
    write_baseline_runtime_config,
)

SGA_ROOT = BASELINE_ROOTS["sga"]


class SGAAdapter:
    def run(self, task, subset, attack, num_iters=None, dry_run=False) -> int:
        if task != "vlr":
            skip(f"SGA 仅支持 VLR，跳过 task={task}", module="baseline")
            return 0

        model = SURROGATE_MODEL.get(task, "albef")
        checkpoint = resolve_checkpoint(task, model)
        if checkpoint is None or not checkpoint.exists():
            if not dry_run:
                skip(f"SGA: checkpoint 不存在 {checkpoint}", module="baseline")
                return 1
            checkpoint = Path("checkpoints/missing.pth")

        rank_dir = SGA_ROOT / "std_eval_idx" / "flickr30k"
        if not dry_run and not (rank_dir / "ALBEF_tr1_rank_index.npy").is_file():
            skip(f"SGA 缺少 rank index: {rank_dir}", module="baseline")
            return 1

        config_path = write_baseline_runtime_config(
            "sga", task, subset, attack, num_iters=num_iters, dry_run=dry_run
        )
        out_dir = output_dir(task, model, "sga", subset)
        result_json = out_dir / "sga_result.json"
        ckpt = path_from_baseline(checkpoint, SGA_ROOT)

        cmd = [
            sys.executable,
            "eval_albef2tcl_flickr.py",
            "--config",
            path_from_baseline(config_path, SGA_ROOT),
            "--source_model",
            "ALBEF",
            "--source_ckpt",
            ckpt,
            "--target_model",
            "ALBEF",
            "--target_ckpt",
            ckpt,
            "--original_rank_index_path",
            "std_eval_idx/flickr30k/",
            "--scales",
            attack.sga_scales,
            "--batch_size",
            str(attack.effective_sga_batch_size_vlr()),
            "--gpu",
            str(attack.gpu),
            "--result_json",
            path_from_baseline(result_json, SGA_ROOT),
            "--save_dir",
            path_from_baseline(out_dir, SGA_ROOT) + "/",
        ]

        header(f"SGA: task={task}, model={model}, subset={subset}")
        if dry_run:
            cmdline(cmd)
            return 0

        reset_output_dir(out_dir)
        rc = run_subprocess(
            cmd,
            cwd=SGA_ROOT,
            env=baseline_env(SGA_ROOT),
            module="baseline",
            label=f"SGA {task}",
        )
        if rc != 0:
            return rc

        if not result_json.is_file():
            skip(f"SGA 未生成结果文件 {result_json}", module="baseline")
            return 1

        try:
            raw = json.loads(result_json.read_text(encoding="utf-8"))
            log_whitebox_baseline(
                "sga", task, model, subset, attack, raw, checkpoint, out_dir
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            skip(f"SGA 结果解析失败: {exc}", module="baseline")
            return 1
        return 0
