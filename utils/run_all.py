import argparse

from .pipeline import run_full, run_quick


def main(argv=None):
    parser = argparse.ArgumentParser(description="实验流水线（非交互）")
    parser.add_argument("--profile", choices=["quick", "smoke", "full"], default="full")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-ablation", action="store_true")
    args = parser.parse_args(argv)

    if args.profile in ("quick", "smoke"):
        run_quick(gpu=args.gpu, dry_run=args.dry_run)
    else:
        run_full(gpu=args.gpu, dry_run=args.dry_run, with_ablation=args.with_ablation)


if __name__ == "__main__":
    main()
