from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
TMM_SCC = ROOT / "tmm_scc"
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = ROOT / "checkpoints"
DATA = ROOT / "data"


@dataclass
class AttackConfig:
    epsilon: float = 12
    num_iters: int = 5
    epsilon_per: float = 0.4
    att_mask: float = 0.1
    kernel_size: int = 5
    momentum: float = 1.0
    num_steps: int = 10
    sim_threshold: float = 0.65
    seed: int = 42
    use_cls: bool = True
    batch_size_vlr: int = 8
    batch_size_ve: int = 8
    gpu: int = 0
    cooldown_seconds: int = 120


@dataclass
class SubsetSpec:
    name: str
    vlr_image_count: int
    ve_entry_count: int
    seed: int = 42


SUBSET_PRESETS: Dict[str, SubsetSpec] = {
    "main_1k": SubsetSpec("main_1k", vlr_image_count=1000, ve_entry_count=1000),
    "ablation_200": SubsetSpec("ablation_200", vlr_image_count=200, ve_entry_count=200),
    "mini_100": SubsetSpec("mini_100", vlr_image_count=100, ve_entry_count=100),
}


DATA_PATHS = {
    "flickr_test_full": DATA / "flickr30k" / "flickr30k_test.json",
    "flickr_image_root": DATA / "flickr30k" / "flickr30k-images",
    "ve_train_full": DATA / "snli-ve" / "ve_train.json",
    "ve_dev_full": DATA / "snli-ve" / "ve_dev.json",
    "ve_test_full": DATA / "snli-ve" / "ve_test.json",
}


def subset_annotation_path(task: str, subset_name: str) -> Path:
    if task == "vlr":
        return DATA / "flickr30k" / f"flickr30k_test_{subset_name}.json"
    if task == "ve":
        return DATA / "snli-ve" / f"ve_test_{subset_name}.json"
    raise ValueError(f"Unknown task: {task}")


CONFIG_TEMPLATES = {
    "vlr": TMM_SCC / "configs" / "Retrieval_flickr.yaml",
    "ve": TMM_SCC / "configs" / "ve_snli-ve.yaml",
}


CLIP_HF_ID = "openai/clip-vit-base-patch16"

HF_VLR_MODELS = {"clip": CLIP_HF_ID}

CHECKPOINT_MAP = {
    "vlr": {
        "albef": CHECKPOINTS / "VLR" / "albef_retrieval_flickr.pth",
        "tcl": CHECKPOINTS / "VLR" / "tcl_retrieval_flickr.pth",
        "blip": CHECKPOINTS / "VLR" / "blip_retrieval_flickr.pth",
    },
    "ve": {
        "albef": CHECKPOINTS / "VE" / "albef_ve_snli_ve.pth",
        "tcl": CHECKPOINTS / "VE" / "tcl_ve_snli_ve.pth",
    },
}


SURROGATE_MODEL = {
    "vlr": "albef",
    "ve": "albef",
}


METHOD_LABELS = {
    "tmm": "TMM",
    "scc": "SCC",
    "coattack": "CoAttack",
    "sga": "SGA",
}


@dataclass
class ExperimentMatrix:
    tasks: List[str] = field(default_factory=lambda: ["vlr", "ve"])
    methods: List[str] = field(default_factory=lambda: ["tmm", "scc"])
    subset: str = "main_1k"
    surrogate: str = "albef"

    def attack_jobs(self) -> List[dict]:
        jobs = []
        for task in self.tasks:
            model = SURROGATE_MODEL.get(task, self.surrogate)
            for method in self.methods:
                if method in ("coattack", "sga"):
                    continue
                jobs.append(
                    {
                        "task": task,
                        "model": model,
                        "method": method,
                        "subset": self.subset,
                    }
                )
        return jobs


def output_dir(task: str, model: str, method: str, subset: str, num_iters: Optional[int] = None) -> Path:
    parts = [OUTPUTS, "whitebox", task, model, method, subset]
    if num_iters is not None:
        parts.append(f"iter_{num_iters}")
    return Path(*parts)


def blackbox_log_path() -> Path:
    return OUTPUTS / "blackbox" / "results_log.jsonl"


def finetune_output_dir(backbone: str) -> Path:
    return OUTPUTS / "finetune" / backbone


def rel_path(path: Path) -> str:
    return "./" + path.relative_to(ROOT).as_posix()


def is_hf_model(task: str, model: str) -> bool:
    return task == "vlr" and model in HF_VLR_MODELS


def resolve_checkpoint(task: str, model: str) -> Optional[Path]:
    if is_hf_model(task, model):
        return None
    candidates = [CHECKPOINT_MAP[task].get(model)]
    legacy = {
        ("vlr", "albef"): CHECKPOINTS / "albef_retrieval_flickr.pth",
        ("vlr", "tcl"): CHECKPOINTS / "tcl_retrieval_flickr.pth",
        ("ve", "albef"): CHECKPOINTS / "albef_ve_snli_ve.pth",
        ("ve", "tcl"): CHECKPOINTS / "tcl_ve_snli_ve.pth",
    }
    candidates.append(legacy.get((task, model)))
    for path in candidates:
        if path and path.exists():
            return path
    return candidates[0]
