from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
TMM_SCC = ROOT / "tmm_scc"
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = ROOT / "checkpoints"
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# 运行环境与输出（改 HF 镜像、日志路径、流水线启动是否清空 outputs 等）
# ---------------------------------------------------------------------------
HF_ENDPOINT = "https://hf-mirror.com"
TEXT_ENCODER = "bert-base-uncased"
SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"
CLEAN_OUTPUT_ON_PIPELINE_START = False  # 已废弃：各任务在 run 前通过 reset_output_dir 单独清理
RUN_LOG = OUTPUTS / "run.json"
GPU = 0


# ---------------------------------------------------------------------------
# 攻击默认参数（VLR batch 须为 5 的倍数：Flickr 每图 5 条 caption）
# 所有白盒方法经 attack_config_for() 取参；控制台改 RunProfile / AttackConfig。
# ---------------------------------------------------------------------------
@dataclass
class AttackConfig:
    epsilon: float = 12
    num_iters: int = 10
    epsilon_per: float = 0.4
    att_mask: float = 0.1
    kernel_size: int = 5
    momentum: float = 1.0
    num_steps: int = 10
    sim_threshold: float = 0.65
    seed: int = 42
    use_cls: bool = True
    batch_size_vlr: int = 5
    batch_size_ve: int = 8
    gpu: int = 0
    cooldown_seconds: int = 120
    sga_batch_size_vlr: Optional[int] = None
    sga_scales: str = "0.5,0.75,1.25,1.5"
    coattack_adv: int = 4

    def effective_sga_batch_size_vlr(self) -> int:
        return (
            self.sga_batch_size_vlr
            if self.sga_batch_size_vlr is not None
            else self.batch_size_vlr
        )


# ---------------------------------------------------------------------------
# 微调
# ---------------------------------------------------------------------------
@dataclass
class FinetuneConfig:
    distill: bool = False
    num_workers: int = 0
    init_from: str = "vlr"
    epochs_default: int = 3


FINETUNE = FinetuneConfig()


# ---------------------------------------------------------------------------
# 数据子集
# ---------------------------------------------------------------------------
@dataclass
class SubsetSpec:
    name: str
    vlr_image_count: int
    ve_entry_count: int
    seed: int = 42


def ve_finetune_counts(spec: SubsetSpec) -> Optional[dict]:
    if spec.name == "main_1k":
        return None
    return {
        "train": max(spec.ve_entry_count * 20, 200),
        "dev": max(spec.ve_entry_count * 4, 40),
        "test": spec.ve_entry_count,
    }


SUBSET_PRESETS: Dict[str, SubsetSpec] = {
    "main_1k": SubsetSpec("main_1k", vlr_image_count=1000, ve_entry_count=1000),
    "ablation_200": SubsetSpec("ablation_200", vlr_image_count=200, ve_entry_count=200),
    "mini_100": SubsetSpec("mini_100", vlr_image_count=100, ve_entry_count=100),
}

# VLR 模板 k_test 默认 64；子集图像/文本数不得小于 k_test（write_runtime_config 会自动收紧）


# ---------------------------------------------------------------------------
# 路径与 checkpoint
# ---------------------------------------------------------------------------
DATA_PATHS = {
    "flickr_test_full": DATA / "flickr30k" / "flickr30k_test.json",
    "flickr_annotation_root": DATA / "flickr30k",
    "flickr_image_root": DATA / "flickr30k" / "flickr30k-images",
    "ve_train_full": DATA / "snli-ve" / "ve_train.json",
    "ve_dev_full": DATA / "snli-ve" / "ve_dev.json",
    "ve_test_full": DATA / "snli-ve" / "ve_test.json",
}


def flickr_image_root_for_task(task: str) -> Path:
    if task == "vlr":
        return DATA_PATHS["flickr_annotation_root"]
    return DATA_PATHS["flickr_image_root"]


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


def clip_local_dir() -> Path:
    return checkpoint_dir("vlr") / "clip" / CLIP_HF_ID.replace("/", "--")


def resolve_clip_checkpoint() -> str:
    local = clip_local_dir()
    if local.is_dir():
        return rel_path(local)
    return f"hf:{CLIP_HF_ID}"


def checkpoint_dir(task: str) -> Path:
    key = task.lower()
    for candidate in (CHECKPOINTS / key, CHECKPOINTS / key.upper()):
        if candidate.is_dir():
            return candidate
    return CHECKPOINTS / key


CHECKPOINT_MAP = {
    "vlr": {
        "albef": checkpoint_dir("vlr") / "albef_retrieval_flickr.pth",
        "tcl": checkpoint_dir("vlr") / "tcl_retrieval_flickr.pth",
        "blip": checkpoint_dir("vlr") / "blip_retrieval_flickr.pth",
    },
    "ve": {
        "albef": checkpoint_dir("ve") / "albef_ve_snli_ve.pth",
        "tcl": checkpoint_dir("ve") / "tcl_ve_snli_ve.pth",
    },
}


SURROGATE_MODEL = {"vlr": "albef", "ve": "albef"}

WHITEBOX_METHODS_BY_TASK: Dict[str, List[str]] = {
    "vlr": ["tmm", "scc", "coattack", "sga"],
    "ve": ["tmm", "scc", "coattack"],
}

WHITEBOX_METHODS = list(dict.fromkeys(m for ms in WHITEBOX_METHODS_BY_TASK.values() for m in ms))

METHOD_LABELS = {
    "tmm": "TMM",
    "scc": "SCC",
    "coattack": "CoAttack",
    "sga": "SGA",
}

BLACKBOX_TARGETS: Dict[str, List[str]] = {
    "vlr": ["tcl", "blip", "clip"],
    "ve": ["tcl"],
}


def resolve_whitebox_methods(task: str, method: str) -> List[str]:
    allowed = WHITEBOX_METHODS_BY_TASK.get(task, [])
    if method == "all":
        return list(allowed)
    if method in allowed:
        return [method]
    return []


def expand_whitebox_method_jobs(tasks: List[str], method: str) -> List[tuple]:
    jobs = []
    for task in tasks:
        for m in resolve_whitebox_methods(task, method):
            jobs.append((task, m))
    return jobs


# ---------------------------------------------------------------------------
# 流水线 profile（控制台「全部重跑」菜单使用）
# ---------------------------------------------------------------------------
@dataclass
class RunProfile:
    name: str
    steps: List[str] = field(default_factory=list)
    subset: str = "main_1k"
    ablation_subset: str = "ablation_200"
    attack_num_iters: Optional[int] = None
    attack_task: str = "all"
    attack_method: str = "all"
    batch_size_vlr: Optional[int] = None
    batch_size_ve: Optional[int] = None
    sga_batch_size_vlr: Optional[int] = None
    sga_scales: Optional[str] = None
    blackbox_task: str = "all"
    blackbox_method: str = "all"
    blackbox_targets: Optional[Dict[str, List[str]]] = None
    ablation_iters: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    cooldown: int = 120
    gpu: int = 0
    finetune_backbone: str = "albef"
    finetune_backbones: Optional[List[str]] = None
    finetune_epochs: Optional[int] = None
    skip_finetune_if_ready: bool = False
    force_finetune: bool = False
    dry_run: bool = False
    clean_outputs_before: bool = False


_PIPELINE_STEPS = ["prepare", "finetune", "attack", "ablation", "blackbox"]

# 本地 smoke：mini_100 全流程（含微调，不跳过）
PIPELINE_TEST_PROFILE = RunProfile(
    name="pipeline_test",
    steps=list(_PIPELINE_STEPS),
    subset="mini_100",
    ablation_subset="mini_100",
    attack_num_iters=2,
    attack_task="all",
    attack_method="all",
    sga_batch_size_vlr=2,
    blackbox_task="all",
    blackbox_method="all",
    blackbox_targets=dict(BLACKBOX_TARGETS),
    ablation_iters=[2, 3],
    cooldown=0,
    finetune_epochs=2,
    finetune_backbones=["albef", "tcl"],
    skip_finetune_if_ready=False,
)

# 正式集：prepare → finetune → attack → blackbox（消融单独跑）
PIPELINE_FULL_PROFILE = RunProfile(
    name="pipeline_full",
    steps=["prepare", "finetune", "attack", "blackbox"],
    subset="main_1k",
    ablation_subset="ablation_200",
    attack_num_iters=10,
    attack_task="all",
    attack_method="all",
    sga_batch_size_vlr=5,
    blackbox_task="all",
    blackbox_method="all",
    blackbox_targets=dict(BLACKBOX_TARGETS),
    ablation_iters=[3, 5, 10, 20],
    cooldown=120,
    finetune_epochs=3,
    finetune_backbones=["albef", "tcl"],
    skip_finetune_if_ready=False,
)

PIPELINE_TEST_FORCE_PROFILE = RunProfile(
    name="pipeline_test_force",
    steps=list(_PIPELINE_STEPS),
    subset="mini_100",
    ablation_subset="mini_100",
    attack_num_iters=2,
    attack_task="all",
    attack_method="all",
    sga_batch_size_vlr=2,
    blackbox_task="all",
    blackbox_method="all",
    blackbox_targets=dict(BLACKBOX_TARGETS),
    ablation_iters=[2, 3],
    cooldown=0,
    finetune_epochs=2,
    finetune_backbones=["albef", "tcl"],
    force_finetune=True,
    skip_finetune_if_ready=False,
)

# ---------------------------------------------------------------------------
# 云端三阶段（推荐顺序：cloud_test → cloud_ablation → cloud_main）
# 正式集运行前 clean_for_cloud_main 保留 run.json 与 ablation_200 消融目录
# ---------------------------------------------------------------------------
CLOUD_TEST_FULL_PROFILE = RunProfile(
    name="cloud_test_full",
    steps=list(_PIPELINE_STEPS),
    subset="mini_100",
    ablation_subset="mini_100",
    attack_num_iters=2,
    attack_task="all",
    attack_method="all",
    batch_size_vlr=5,
    batch_size_ve=8,
    sga_batch_size_vlr=5,
    blackbox_task="all",
    blackbox_method="all",
    blackbox_targets=dict(BLACKBOX_TARGETS),
    ablation_iters=[2, 3, 5],
    cooldown=60,
    finetune_epochs=2,
    finetune_backbones=["albef", "tcl"],
    skip_finetune_if_ready=False,
)

CLOUD_ABLATION_PROFILE = RunProfile(
    name="cloud_ablation",
    steps=["prepare", "ablation"],
    subset="ablation_200",
    ablation_subset="ablation_200",
    ablation_iters=[3, 5, 10, 20],
    cooldown=120,
    gpu=0,
)

CLOUD_MAIN_FULL_PROFILE = RunProfile(
    name="cloud_main_full",
    steps=["prepare", "finetune", "attack", "blackbox"],
    subset="main_1k",
    ablation_subset="ablation_200",
    attack_num_iters=10,
    attack_task="all",
    attack_method="all",
    batch_size_vlr=10,
    batch_size_ve=16,
    sga_batch_size_vlr=10,
    blackbox_task="all",
    blackbox_method="all",
    blackbox_targets=dict(BLACKBOX_TARGETS),
    cooldown=120,
    finetune_epochs=3,
    finetune_backbones=["albef", "tcl"],
    skip_finetune_if_ready=False,
    clean_outputs_before=True,
)

CLOUD_MAIN_NO_FINETUNE_PROFILE = RunProfile(
    name="cloud_main_no_finetune",
    steps=["prepare", "attack", "blackbox"],
    subset="main_1k",
    attack_num_iters=10,
    attack_task="all",
    attack_method="all",
    batch_size_vlr=10,
    batch_size_ve=16,
    sga_batch_size_vlr=10,
    blackbox_task="all",
    blackbox_method="all",
    blackbox_targets=dict(BLACKBOX_TARGETS),
    cooldown=120,
    skip_finetune_if_ready=True,
    clean_outputs_before=True,
)


def profile_with_gpu(profile: RunProfile) -> RunProfile:
    return replace(profile, gpu=GPU)


PROFILES: Dict[str, RunProfile] = {
    "pipeline_test": PIPELINE_TEST_PROFILE,
    "pipeline_full": PIPELINE_FULL_PROFILE,
    "pipeline_test_force": PIPELINE_TEST_FORCE_PROFILE,
    "cloud_test_full": CLOUD_TEST_FULL_PROFILE,
    "cloud_ablation": CLOUD_ABLATION_PROFILE,
    "cloud_main_full": CLOUD_MAIN_FULL_PROFILE,
    "cloud_main_no_finetune": CLOUD_MAIN_NO_FINETUNE_PROFILE,
}


def get_profile(name: str) -> RunProfile:
    if name not in PROFILES:
        raise KeyError(f"未知 profile: {name}，可选: {list(PROFILES)}")
    return PROFILES[name]


def attack_config_for(
    profile: Optional[RunProfile] = None,
    *,
    gpu: Optional[int] = None,
    num_iters: Optional[int] = None,
) -> AttackConfig:
    """由 RunProfile 生成攻击参数；控制台 / pipeline 均走此函数。"""
    device = profile.gpu if profile is not None else (GPU if gpu is None else gpu)
    cfg = replace(AttackConfig(), gpu=device)
    if profile is not None:
        if profile.attack_num_iters is not None:
            cfg = replace(cfg, num_iters=profile.attack_num_iters)
        if profile.batch_size_vlr is not None:
            cfg = replace(cfg, batch_size_vlr=profile.batch_size_vlr)
        if profile.batch_size_ve is not None:
            cfg = replace(cfg, batch_size_ve=profile.batch_size_ve)
        if profile.sga_batch_size_vlr is not None:
            cfg = replace(cfg, sga_batch_size_vlr=profile.sga_batch_size_vlr)
        if profile.sga_scales is not None:
            cfg = replace(cfg, sga_scales=profile.sga_scales)
    if num_iters is not None:
        cfg = replace(cfg, num_iters=num_iters)
    return cfg


HARDWARE_PRESETS: Dict[str, Dict[str, object]] = {
    "local_8g": {
        "sga_batch_size_vlr": 2,
        "batch_size_vlr": 5,
        "batch_size_ve": 8,
    },
    "server_24g": {
        "sga_batch_size_vlr": 10,
        "batch_size_vlr": 10,
        "batch_size_ve": 16,
    },
}


def apply_hardware(profile: RunProfile, preset: str) -> RunProfile:
    opts = HARDWARE_PRESETS.get(preset, {})
    return replace(profile, **opts)


def apply_runtime_env(env: dict) -> dict:
    env.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    return env


def run_log_path() -> Path:
    return RUN_LOG


def blackbox_log_path() -> Path:
    return run_log_path()


# ---------------------------------------------------------------------------
# 实验矩阵与输出目录
# ---------------------------------------------------------------------------
@dataclass
class ExperimentMatrix:
    tasks: List[str] = field(default_factory=lambda: ["vlr", "ve"])
    method: str = "all"
    subset: str = "main_1k"
    surrogate: str = "albef"

    def attack_jobs(self) -> List[dict]:
        jobs = []
        for task in self.tasks:
            model = SURROGATE_MODEL.get(task, self.surrogate)
            for method in resolve_whitebox_methods(task, self.method):
                jobs.append(
                    {
                        "task": task,
                        "model": model,
                        "method": method,
                        "subset": self.subset,
                    }
                )
        return jobs


def output_dir(
    task: str, model: str, method: str, subset: str, num_iters: Optional[int] = None
) -> Path:
    parts = [OUTPUTS, "whitebox", task, model, method, subset]
    if num_iters is not None:
        parts.append(f"iter_{num_iters}")
    return Path(*parts)


def finetune_output_dir(backbone: str) -> Path:
    return OUTPUTS / "finetune" / backbone


def rel_path(path: Path) -> str:
    return "./" + path.relative_to(ROOT).as_posix()


def third_party_rel(repo: Path, path: Path) -> str:
    import os

    return os.path.relpath(path.resolve(), repo.resolve()).replace("\\", "/")


def is_hf_model(task: str, model: str) -> bool:
    return task == "vlr" and model in HF_VLR_MODELS


def resolve_checkpoint(task: str, model: str) -> Optional[Path]:
    if is_hf_model(task, model):
        return None
    expected = CHECKPOINT_MAP.get(task, {}).get(model)
    if expected is None:
        return None
    if expected.exists():
        return expected
    legacy_names = {
        ("vlr", "albef"): "albef_retrieval_flickr.pth",
        ("vlr", "tcl"): "tcl_retrieval_flickr.pth",
        ("vlr", "blip"): "model_base_retrieval_flickr.pth",
        ("ve", "albef"): "albef_ve_snli_ve.pth",
        ("ve", "tcl"): "tcl_ve_snli_ve.pth",
    }
    legacy_name = legacy_names.get((task, model))
    if legacy_name:
        for base in (CHECKPOINTS, checkpoint_dir(task)):
            legacy = base / legacy_name
            if legacy.exists():
                return legacy
    return expected
