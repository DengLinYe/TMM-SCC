def run_baseline_attack(
    task: str,
    method: str,
    subset: str,
    attack,
    num_iters=None,
    dry_run: bool = False,
) -> int:
    if method == "coattack":
        from .co_attack.adapter import CoAttackAdapter

        adapter = CoAttackAdapter()
    elif method == "sga":
        from .sga.adapter import SGAAdapter

        adapter = SGAAdapter()
    else:
        raise ValueError(f"Unknown baseline: {method}")

    return adapter.run(
        task=task,
        subset=subset,
        attack=attack,
        num_iters=num_iters,
        dry_run=dry_run,
    )
