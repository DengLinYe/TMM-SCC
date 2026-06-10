from dataclasses import replace

from .config import (
    BLACKBOX_TARGETS,
    GPU,
    SUBSET_PRESETS,
    RunProfile,
    apply_hardware,
)
from .log import ok, warn
from .pipeline import (
    run_cloud_ablation,
    run_cloud_main_full,
    run_cloud_main_no_finetune,
    run_cloud_test_full,
    run_pipeline,
    run_pipeline_full,
    run_pipeline_test,
    run_pipeline_test_force,
)


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or default


def _prompt_int(text: str, default: int) -> int:
    raw = _prompt(text, str(default))
    try:
        return int(raw)
    except ValueError:
        warn(f"无效输入，使用默认 {default}", module="console")
        return default


def _prompt_choice(text: str, choices: list, default: str) -> str:
    opts = "/".join(choices)
    while True:
        value = _prompt(f"{text} ({opts})", default)
        if value in choices:
            return value
        warn(f"请输入 {opts} 之一", module="console")


def _print_menu():
    print("\n" + "=" * 50)
    print("  TMM-SCC 实验控制台")
    print(f"  GPU = {GPU}  （修改 utils/config.py 中的 GPU）")
    print("=" * 50)
    print("  分步运行")
    print("  1. 生成数据子集")
    print("  2. 白盒攻击  (VLR: TMM/SCC/Co-Attack/SGA; VE: TMM/SCC/Co-Attack)")
    print("  3. 黑盒评测  (按各白盒方法的 adv 样本迁移到受害模型)")
    print("  4. num_iters 消融  (VE, TMM vs SCC)")
    print("  5. VE 微调 (ALBEF / TCL)")
    print("  本地 smoke")
    print("  6. 测试全流程  mini_100 / iters=2 / 含微调与消融")
    print("  8. 强制测试    同 6，但强制重跑 VE 微调")
    print("  云端三阶段（推荐顺序 9 → 10 → 11）")
    print("  9.  云端·测试全流程   mini_100 / 24G 预设")
    print("  10. 云端·消融实验     ablation_200 / 含 prepare / 选定正式 num_iters")
    print("  11. 云端·正式全流程   main_1k / iters=10 / 无消融 / 清 outputs")
    print("  12. 云端·正式(无微调) main_1k / 复用已有 VE 权重")
    print("  7.  正式全流程(本地)  main_1k / iters=10 / 无消融")
    print("  q. 退出")
    print("=" * 50)
    print("  参数: utils/config.py → AttackConfig / RunProfile / HARDWARE_PRESETS")
    print("=" * 50)


def _base_profile() -> RunProfile:
    return RunProfile(name="interactive", gpu=GPU, blackbox_targets=dict(BLACKBOX_TARGETS))


def _menu_prepare():
    print("\n可用预设:", ", ".join(SUBSET_PRESETS.keys()), ", 或自定义名称")
    name = _prompt("子集名称", "main_1k")
    if name not in SUBSET_PRESETS:
        vlr = _prompt_int("VLR 图像数", 1000)
        ve = _prompt_int("VE 条目数", 1000)
        from .prepare_subset import create_subset
        from .config import SubsetSpec

        create_subset(SubsetSpec(name, vlr, ve))
        ok(f"子集 {name} 已生成", module="console")
    else:
        run_pipeline(["prepare"], replace(_base_profile(), name="prepare", subset=name))


def _menu_attack():
    p = _base_profile()
    p.subset = _prompt("子集", "main_1k")
    p.attack_task = _prompt_choice("任务", ["vlr", "ve", "all"], "all")
    p.attack_method = _prompt_choice(
        "方法", ["tmm", "scc", "coattack", "sga", "all"], "all"
    )
    hw = _prompt_choice("显存环境", ["local_8g", "server_24g"], "local_8g")
    p = apply_hardware(p, hw)
    iters_raw = _prompt("attack_num_iters (留空=AttackConfig 默认 10)", "")
    p.attack_num_iters = int(iters_raw) if iters_raw else None
    p.cooldown = _prompt_int("任务间隔秒数", 120)
    run_pipeline(["attack"], p)


def _menu_blackbox():
    p = _base_profile()
    p.subset = _prompt("子集", "main_1k")
    p.blackbox_task = _prompt_choice("任务", ["vlr", "ve", "all"], "all")
    p.blackbox_method = _prompt_choice(
        "方法", ["tmm", "scc", "coattack", "sga", "all"], "all"
    )
    targets_raw = _prompt("受害模型 (空格分隔)", "tcl blip clip")
    targets = targets_raw.split()
    p.blackbox_targets = {
        "vlr": targets,
        "ve": [t for t in targets if t in BLACKBOX_TARGETS["ve"]] or ["tcl"],
    }
    run_pipeline(["blackbox"], p)


def _menu_ablation():
    p = _base_profile()
    p.ablation_subset = _prompt("消融子集", "ablation_200")
    iters_raw = _prompt("iters 列表 (空格分隔)", "3 5 10 20")
    p.ablation_iters = [int(x) for x in iters_raw.split()]
    p.cooldown = _prompt_int("任务间隔秒数", 120)
    run_pipeline(["ablation"], p)


def _menu_finetune():
    p = _base_profile()
    p.finetune_backbone = _prompt_choice("backbone", ["albef", "tcl"], "albef")
    p.subset = _prompt("测试集子集名", "main_1k")
    p.finetune_epochs = _prompt_int("epochs", 3)
    run_pipeline(["finetune"], p)


def _cloud_hw() -> str:
    return _prompt_choice("显存环境", ["local_8g", "server_24g"], "server_24g")


def run_console():
    while True:
        _print_menu()
        choice = input("请选择: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("再见。")
            break
        if choice == "1":
            _menu_prepare()
        elif choice == "2":
            _menu_attack()
        elif choice == "3":
            _menu_blackbox()
        elif choice == "4":
            _menu_ablation()
        elif choice == "5":
            _menu_finetune()
        elif choice == "6":
            run_pipeline_test()
        elif choice == "7":
            run_pipeline_full()
        elif choice == "8":
            run_pipeline_test_force()
        elif choice == "9":
            run_cloud_test_full(_cloud_hw())
        elif choice == "10":
            run_cloud_ablation(_cloud_hw())
        elif choice == "11":
            run_cloud_main_full(_cloud_hw())
        elif choice == "12":
            run_cloud_main_no_finetune(_cloud_hw())
        else:
            warn("无效选项", module="console")


def main():
    run_console()
