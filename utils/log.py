"""统一控制台输出与错误处理。"""

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Union

Cmd = Union[str, Iterable[str]]

_LEVEL_STREAM = {
    "WARN": sys.stderr,
    "ERROR": sys.stderr,
    "SKIP": sys.stderr,
}


class PipelineError(Exception):
    pass


def _emit(level: str, msg: str, module: Optional[str] = None) -> None:
    tag = f"[{module}]" if module else ""
    parts = [f"{level:<5}", tag, msg]
    line = " ".join(p for p in parts if p)
    print(line, file=_LEVEL_STREAM.get(level, sys.stdout))


def info(msg: str, module: Optional[str] = None) -> None:
    _emit("INFO", msg, module)


def ok(msg: str, module: Optional[str] = None) -> None:
    _emit("OK", msg, module)


def warn(msg: str, module: Optional[str] = None) -> None:
    _emit("WARN", msg, module)


def error(msg: str, module: Optional[str] = None) -> None:
    _emit("ERROR", msg, module)


def skip(reason: str, module: Optional[str] = None) -> None:
    _emit("SKIP", f"跳过: {reason}", module)


def step(msg: str, module: Optional[str] = None) -> None:
    _emit("RUN", msg, module)


def wait(msg: str, module: Optional[str] = None) -> None:
    _emit("WAIT", msg, module)


def header(title: str, width: int = 50, char: str = "=") -> None:
    print(f"\n{char * width}\n{title}\n{char * width}")


def banner(title: str, width: int = 60, char: str = "#") -> None:
    print(f"\n{char * width}\n# {title}\n{char * width}")


def cmdline(cmd: Cmd) -> None:
    if isinstance(cmd, str):
        print(cmd)
    else:
        print(" ".join(cmd))


def die(msg: str, code: int = 1, module: Optional[str] = None) -> NoReturn:
    error(msg, module)
    sys.exit(code)


def finish_ok(msg: str, module: Optional[str] = None) -> None:
    ok(msg, module)


def finish_fail(failed: int, label: str = "任务", module: Optional[str] = None) -> NoReturn:
    warn(f"完成，{failed} 个{label}失败", module)
    sys.exit(1)


def require_file(path: Path, label: str, module: Optional[str] = None) -> Path:
    if not path.is_file():
        die(f"{label} 不存在: {path}", module=module)
    return path


def require_dir(path: Path, label: str, module: Optional[str] = None) -> Path:
    if not path.is_dir():
        die(f"{label} 不存在: {path}", module=module)
    return path


def run_subprocess(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    module: Optional[str] = None,
    label: Optional[str] = None,
) -> int:
    if label:
        step(label, module=module)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    except FileNotFoundError as exc:
        die(f"无法启动子进程: {exc}", module=module)
    if result.returncode != 0:
        hint = (
            f"子进程异常退出 (code={result.returncode})\n"
            f"  命令: {' '.join(cmd)}\n"
            f"  工作目录: {cwd or Path.cwd()}\n"
            "  若为 CUDA 索引越界，常见原因是上次运行残留状态导致初始化路径变化；"
            "请查看该进程上方的 Python/CUDA 日志。"
        )
        error(hint, module=module)
        return result.returncode
    return 0
