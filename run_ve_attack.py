import os
import subprocess
import sys
import time


def run_final_experiments():
    cooldown_seconds = 300

    os.makedirs("./output/VE_SCC/final", exist_ok=True)
    os.makedirs("./output/VE_TMM/final", exist_ok=True)

    cmd_scc = [
        sys.executable,
        "EvalVEAttack.py",
        "--adv",
        "1",
        "--gpu",
        "0",
        "--checkpoint",
        "./checkpoints/albef_ve_snli_ve.pth",
        "--config",
        "./configs/ve_snli-ve.yaml",
        "--save_json_name",
        "result.json",
        "--config_name",
        "run_config",
        "--text_method",
        "scc",
        "--sim_threshold",
        "0.65",
        "--output_dir",
        "./output/VE_SCC/final",
        "--save_dir",
        "./output/VE_SCC/final/",
        "--log_name",
        "result",
    ]

    cmd_tmm = [
        sys.executable,
        "EvalVEAttack.py",
        "--adv",
        "1",
        "--gpu",
        "0",
        "--checkpoint",
        "./checkpoints/albef_ve_snli_ve.pth",
        "--config",
        "./configs/ve_snli-ve.yaml",
        "--save_json_name",
        "result.json",
        "--config_name",
        "run_config",
        "--text_method",
        "tmm",
        "--output_dir",
        "./output/VE_TMM/final",
        "--save_dir",
        "./output/VE_TMM/final/",
        "--log_name",
        "result",
    ]

    print(">>> 开始执行最终实验: SCC 攻击")
    subprocess.run(cmd_scc)

    print(f"\n>>> SCC 执行完毕。系统休眠 {cooldown_seconds} 秒以降低 GPU 温度...\n")
    time.sleep(cooldown_seconds)

    print(">>> 开始执行最终实验: TMM 原版攻击")
    subprocess.run(cmd_tmm)

    print("\n>>> 所有最终实验执行完毕！")


if __name__ == "__main__":
    run_final_experiments()
