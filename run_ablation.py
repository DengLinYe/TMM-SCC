import os
import subprocess
import sys
import time

from ruamel.yaml import YAML


def modify_yaml_iters(yaml_path, new_iters):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    config["num_iters"] = new_iters

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    print(f"\n[+] 已将 {yaml_path} 中的 num_iters 修改为: {new_iters}")


def run_experiment():
    yaml_path = "./configs/ve_snli-ve.yaml"
    iters_list = [1, 3, 5, 10]
    cooldown_seconds = 120

    for iters in iters_list:
        print(f"\n{'=' * 50}")
        print(f"🚀 开始执行消融实验: num_iters = {iters}")
        print(f"{'=' * 50}")

        modify_yaml_iters(yaml_path, iters)

        scc_out_dir = f"./output/VE_SCC/iter_{iters}"
        tmm_out_dir = f"./output/VE_TMM/iter_{iters}"

        os.makedirs(scc_out_dir, exist_ok=True)
        os.makedirs(tmm_out_dir, exist_ok=True)

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
            yaml_path,
            "--save_json_name",
            "result.json",
            "--config_name",
            "run_config.yaml",
            "--text_method",
            "scc",
            "--sim_threshold",
            "0.65",
            "--output_dir",
            scc_out_dir,
            "--save_dir",
            scc_out_dir,
            "--log_name",
            "result.txt",
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
            yaml_path,
            "--save_json_name",
            "result.json",
            "--config_name",
            "run_config.yaml",
            "--text_method",
            "tmm",
            "--output_dir",
            tmm_out_dir,
            "--save_dir",
            tmm_out_dir,
            "--log_name",
            "result.txt",
        ]

        print(f"\n[▶] 正在运行 SCC 攻击 (iters={iters})...")
        subprocess.run(cmd_scc)

        print(f"\n[⏳] SCC 运行完毕，休眠 {cooldown_seconds} 秒让显卡散热...")
        time.sleep(cooldown_seconds)

        print(f"\n[▶] 正在运行 TMM 原版攻击 (iters={iters})...")
        subprocess.run(cmd_tmm)

        print(f"\n[⏳] TMM 运行完毕，休眠 {cooldown_seconds} 秒让显卡散热...")
        time.sleep(cooldown_seconds)

    print("\n🎉 所有消融实验执行完毕！")


if __name__ == "__main__":
    run_experiment()
