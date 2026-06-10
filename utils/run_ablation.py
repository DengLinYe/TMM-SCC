import argparse
import time

from .config import RunProfile, attack_config_for
from .log import finish_fail, finish_ok, info
from .run_attack import run_single_attack

DEFAULT_ITERS = [3, 5, 10, 20]


def main(argv=None, profile: RunProfile = None):
    parser = argparse.ArgumentParser(description="num_iters 消融实验（VE，TMM vs SCC）")
    parser.add_argument("--subset", default="ablation_200")
    parser.add_argument("--iters", type=int, nargs="+", default=DEFAULT_ITERS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cooldown", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if profile is not None:
        attack = attack_config_for(profile, gpu=profile.gpu)
        subset = profile.ablation_subset
        iters_list = profile.ablation_iters
        cooldown = profile.cooldown
        dry_run = profile.dry_run or args.dry_run
    else:
        attack = attack_config_for(gpu=args.gpu)
        subset = args.subset
        iters_list = args.iters
        cooldown = args.cooldown
        dry_run = args.dry_run

    methods = ["scc", "tmm"]
    failed = 0
    total = len(iters_list) * len(methods)
    idx = 0

    for iters in iters_list:
        for method in methods:
            idx += 1
            info(f"消融 {idx}/{total}: num_iters={iters}, method={method}", module="ablation")
            rc = run_single_attack(
                task="ve",
                model="albef",
                method=method,
                subset=subset,
                attack=attack,
                num_iters=iters,
                ablation=True,
                dry_run=dry_run,
            )
            if rc != 0:
                failed += 1
            elif not dry_run and idx < total:
                time.sleep(cooldown)

    if failed:
        finish_fail(failed, label="消融", module="ablation")
    finish_ok("消融实验全部完成", module="ablation")
