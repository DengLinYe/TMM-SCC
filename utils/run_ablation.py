import argparse
import sys
import time

from .config import AttackConfig
from .run_attack import run_single_attack

DEFAULT_ITERS = [3, 5, 10, 20]


def main(argv=None):
    parser = argparse.ArgumentParser(description="num_iters 消融实验（VE，TMM vs SCC）")
    parser.add_argument("--subset", default="ablation_200")
    parser.add_argument("--iters", type=int, nargs="+", default=DEFAULT_ITERS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cooldown", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    attack = AttackConfig(gpu=args.gpu)
    methods = ["scc", "tmm"]
    failed = 0
    total = len(args.iters) * len(methods)
    step = 0

    for iters in args.iters:
        for method in methods:
            step += 1
            print(f"\n[消融 {step}/{total}] num_iters={iters}, method={method}")
            rc = run_single_attack(
                task="ve",
                model="albef",
                method=method,
                subset=args.subset,
                attack=attack,
                num_iters=iters,
                dry_run=args.dry_run,
            )
            if rc != 0:
                failed += 1
            elif not args.dry_run and step < total:
                time.sleep(args.cooldown)

    if failed:
        print(f"\n[!] 消融完成，{failed} 个任务失败")
        sys.exit(1)
    print("\n[+] 消融实验全部完成")
