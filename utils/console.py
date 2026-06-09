import sys

from .config import SUBSET_PRESETS
from .pipeline import FULL_PROFILE, QUICK_PROFILE, RunProfile, run_full, run_pipeline, run_quick


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or default


def _prompt_int(text: str, default: int) -> int:
    raw = _prompt(text, str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"[!] 无效输入，使用默认 {default}")
        return default


def _prompt_choice(text: str, choices: list, default: str) -> str:
    opts = "/".join(choices)
    while True:
        value = _prompt(f"{text} ({opts})", default)
        if value in choices:
            return value
        print(f"[!] 请输入 {opts} 之一")


def _print_menu():
    p = QUICK_PROFILE
    print("\n" + "=" * 50)
    print("  TMM-SCC 实验控制台")
    print("=" * 50)
    print(f"  0. 快速验证全流程")
    print(f"     ({p.subset}, num_iters={p.num_iters}, epochs={p.finetune_epochs},")
    print(f"      TMM+SCC, 黑盒 tcl+clip, 含微调+消融)")
    print("  1. 生成数据子集")
    print("  2. 白盒攻击")
    print("  3. 黑盒评测")
    print("  4. num_iters 消融")
    print("  5. VE 微调 (ALBEF / TCL)")
    print("  6. 正式实验流水线 (main_1k)")
    print("  q. 退出")
    print("=" * 50)


def _base_profile(gpu: int) -> RunProfile:
    return RunProfile(name="interactive", gpu=gpu)


def _menu_prepare(gpu: int):
    print("\n可用预设:", ", ".join(SUBSET_PRESETS.keys()), ", 或自定义名称")
    name = _prompt("子集名称", "main_1k")
    if name not in SUBSET_PRESETS:
        vlr = _prompt_int("VLR 图像数", 1000)
        ve = _prompt_int("VE 条目数", 1000)
        from .prepare_subset import create_subset
        from .config import SubsetSpec

        create_subset(SubsetSpec(name, vlr, ve))
        print(f"[+] 子集 {name} 已生成")
    else:
        run_pipeline(["prepare"], RunProfile(name="prepare", subset=name, gpu=gpu))


def _menu_attack(gpu: int):
    p = _base_profile(gpu)
    p.subset = _prompt("子集", "main_1k")
    p.attack_task = _prompt_choice("任务", ["vlr", "ve", "all"], "all")
    p.attack_method = _prompt_choice("方法", ["tmm", "scc", "all"], "all")
    iters_raw = _prompt("num_iters (留空=配置默认)", "")
    p.num_iters = int(iters_raw) if iters_raw else None
    p.cooldown = _prompt_int("任务间隔秒数", 120)
    run_pipeline(["attack"], p)


def _menu_blackbox(gpu: int):
    p = _base_profile(gpu)
    p.subset = _prompt("子集", "main_1k")
    p.blackbox_task = _prompt_choice("任务", ["vlr", "ve", "all"], "all")
    p.blackbox_method = _prompt_choice("方法", ["tmm", "scc", "all"], "all")
    targets_raw = _prompt("受害模型 (空格分隔)", "tcl clip")
    targets = targets_raw.split()
    p.blackbox_targets = {
        "vlr": targets,
        "ve": [t for t in targets if t != "clip"] or ["tcl"],
    }
    run_pipeline(["blackbox"], p)


def _menu_ablation(gpu: int):
    p = _base_profile(gpu)
    p.ablation_subset = _prompt("消融子集", "ablation_200")
    iters_raw = _prompt("iters 列表 (空格分隔)", "3 5 10 20")
    p.ablation_iters = [int(x) for x in iters_raw.split()]
    p.cooldown = _prompt_int("任务间隔秒数", 120)
    run_pipeline(["ablation"], p)


def _menu_finetune(gpu: int):
    p = _base_profile(gpu)
    p.finetune_backbone = _prompt_choice("backbone", ["albef", "tcl"], "albef")
    p.subset = _prompt("测试集子集名", "main_1k")
    p.finetune_epochs = _prompt_int("epochs", 3)
    run_pipeline(["finetune"], p)


def _menu_full(gpu: int):
    p = RunProfile(**{**FULL_PROFILE.__dict__, "gpu": gpu})
    p.subset = _prompt("子集", "main_1k")
    with_ab = _prompt_choice("是否包含消融", ["y", "n"], "n") == "y"
    run_full(gpu=gpu, with_ablation=with_ab)


def run_console():
    gpu = _prompt_int("GPU 编号", 0)
    while True:
        _print_menu()
        choice = input("请选择: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("再见。")
            break
        if choice == "0":
            run_quick(gpu=gpu)
        elif choice == "1":
            _menu_prepare(gpu)
        elif choice == "2":
            _menu_attack(gpu)
        elif choice == "3":
            _menu_blackbox(gpu)
        elif choice == "4":
            _menu_ablation(gpu)
        elif choice == "5":
            _menu_finetune(gpu)
        elif choice == "6":
            _menu_full(gpu)
        else:
            print("[!] 无效选项")


def _parse_cli_gpu(argv: list) -> int:
    if "--gpu" in argv:
        return int(argv[argv.index("--gpu") + 1])
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    dry = "--dry-run" in argv

    if "--quick" in argv or "--smoke" in argv:
        run_quick(gpu=_parse_cli_gpu(argv), dry_run=dry)
        return
    if "--full" in argv:
        run_full(
            gpu=_parse_cli_gpu(argv),
            dry_run=dry,
            with_ablation="--with-ablation" in argv,
        )
        return
    run_console()


if __name__ == "__main__":
    main()
