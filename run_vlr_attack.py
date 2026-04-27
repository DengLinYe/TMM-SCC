import os
import subprocess
import sys
import time


def run_final_experiments():
    cooldown_seconds = 300

    scc_out_dir = "./output/VLR_SCC"
    tmm_out_dir = "./output/VLR_TMM"

    os.makedirs(scc_out_dir, exist_ok=True)
    os.makedirs(tmm_out_dir, exist_ok=True)

    cmd_scc = [
        sys.executable,
        "EvalTransferAttack.py",
        "--adv",
        "1",
        "--gpu",
        "0",
        "--checkpoint",
        "./checkpoints/albef_retrieval_flickr.pth",
        "--config",
        "./configs/Retrieval_flickr.yaml",
        "--save_json_name",
        "result.json",
        "--config_name",
        "run_config",
        "--text_method",
        "scc",
        "--sim_threshold",
        "0.65",
        "--output_dir",
        scc_out_dir,
        "--save_dir",
        f"{scc_out_dir}/",
        "--log_name",
        "result",
    ]

    cmd_tmm = [
        sys.executable,
        "EvalTransferAttack.py",
        "--adv",
        "1",
        "--gpu",
        "0",
        "--checkpoint",
        "./checkpoints/albef_retrieval_flickr.pth",
        "--config",
        "./configs/Retrieval_flickr.yaml",
        "--save_json_name",
        "result.json",
        "--config_name",
        "run_config",
        "--text_method",
        "tmm",
        "--output_dir",
        tmm_out_dir,
        "--save_dir",
        f"{tmm_out_dir}/",
        "--log_name",
        "result",
    ]

    print(">>> 开始执行 VLR 最终实验: SCC 攻击 (ALBEF)")
    subprocess.run(cmd_scc)

    print(f"\n>>> SCC 执行完毕。系统休眠 {cooldown_seconds} 秒以降低 GPU 温度...\n")
    time.sleep(cooldown_seconds)

    print(">>> 开始执行 VLR 最终实验: TMM 原版攻击 (ALBEF)")
    subprocess.run(cmd_tmm)

    print(f"\n>>> TMM 执行完毕。系统休眠 {cooldown_seconds} 秒以降低 GPU 温度...\n")
    time.sleep(cooldown_seconds)

    print("\n>>> 所有最终实验执行完毕！")


def shutdown_system():
    print("\n[!] 实验全部完成，系统将在 60 秒后关机。")
    if os.name == "nt":
        os.system("shutdown /s /t 60")
    else:
        os.system("shutdown -h +1")


if __name__ == "__main__":
    run_final_experiments()
    shutdown_system()
