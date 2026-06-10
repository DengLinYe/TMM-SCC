"""受害模型（TCL / BLIP / ALBEF）权重加载：按 checkpoint 推断结构差异。"""

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from models.model_retrieval import ALBEF as RetrievalModel
from models.model_ve import ALBEF as VEModel
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed


def _remap_bert_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, val in state_dict.items():
        if key.startswith("bert."):
            out[key.replace("bert.", "", 1)] = val
        else:
            out[key] = val
    return out


def init_blip_tokenizer() -> BertTokenizer:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
    return tokenizer


def _infer_overrides(
    state_dict: Dict[str, torch.Tensor], target: str
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if "image_queue" in state_dict:
        overrides["queue_size"] = int(state_dict["image_queue"].shape[1])
    if target == "blip":
        for key in (
            "text_encoder.embeddings.word_embeddings.weight",
            "bert.embeddings.word_embeddings.weight",
        ):
            if key in state_dict:
                overrides["vocab_size"] = int(state_dict[key].shape[0])
                break
        if any("text_encoder.encoder.layer.0.crossattention" in k for k in state_dict):
            overrides["fusion_layer"] = 0
    return overrides


def _co_attack_state_dict(
    state_dict: Dict[str, torch.Tensor], target: str
) -> Dict[str, torch.Tensor]:
    """与 Co-Attack RetrievalEval 一致：仅 remap bert. 前缀，不做 fusion/vocab 魔改。"""
    sd = dict(state_dict)
    if target == "blip":
        for key in list(sd.keys()):
            if key.startswith("bert."):
                sd[key.replace("bert.", "", 1)] = sd.pop(key)
        if "ptr_queue" in sd and "queue_ptr" not in sd:
            sd["queue_ptr"] = sd.pop("ptr_queue")
    elif target == "tcl":
        sd = _remap_bert_keys(sd)
    return sd


def _resolve_bert_config_path(base_path: str, bert_overrides: Optional[Dict[str, Any]]) -> str:
    if not bert_overrides:
        return base_path
    with open(base_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg_dict.update(bert_overrides)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(cfg_dict, tmp)
    tmp.close()
    return tmp.name


def _normalize_state_dict(
    state_dict: Dict[str, torch.Tensor], target: str
) -> Dict[str, torch.Tensor]:
    sd = dict(state_dict)
    if target == "blip" and "ptr_queue" in sd and "queue_ptr" not in sd:
        sd["queue_ptr"] = sd["ptr_queue"]
    if target == "tcl":
        sd = _remap_bert_keys(sd)
    return sd


def _filter_compatible(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    resized = []
    for key, val in state_dict.items():
        if key not in model_sd:
            continue
        if model_sd[key].shape != val.shape:
            if (
                key.endswith("embeddings.word_embeddings.weight")
                and val.ndim == 2
                and model_sd[key].ndim == 2
                and val.shape[1] == model_sd[key].shape[1]
            ):
                rows = min(val.shape[0], model_sd[key].shape[0])
                adapted = model_sd[key].clone()
                adapted[:rows] = val[:rows]
                filtered[key] = adapted
                resized.append(f"{key}: {tuple(val.shape)} -> {tuple(adapted.shape)}")
                continue
            skipped.append(key)
            continue
        filtered[key] = val
    if resized:
        preview = ", ".join(resized[:4])
        more = f" ...(+{len(resized) - 4})" if len(resized) > 4 else ""
        print(f"适配词表权重 {len(resized)} 个: {preview}{more}")
    if skipped:
        preview = ", ".join(skipped[:6])
        more = f" ...(+{len(skipped) - 6})" if len(skipped) > 6 else ""
        print(f"跳过 shape 不匹配的权重 {len(skipped)} 个: {preview}{more}")
    return filtered


def load_victim_checkpoint(
    task: str,
    config: dict,
    checkpoint_path: str,
    device: torch.device,
    target: str = "tcl",
) -> Tuple[torch.nn.Module, BertTokenizer]:
    root = Path(__file__).resolve().parent.parent

    def _resolve(path: str) -> str:
        if not path:
            return path
        p = Path(path)
        if p.is_absolute():
            return str(p)
        if path.startswith("./"):
            return str((root / path[2:]).resolve())
        return str((root / path).resolve())

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    cfg = copy.deepcopy(config)
    cfg["bert_config"] = _resolve(cfg["bert_config"])
    overrides = _infer_overrides(state_dict, target)
    if "queue_size" in overrides:
        cfg["queue_size"] = overrides["queue_size"]
    if overrides.get("vocab_size") and overrides["vocab_size"] != 30522:
        cfg["init_text_encoder_from_pretrained"] = False

    bert_overrides = {
        k: overrides[k] for k in ("vocab_size", "fusion_layer") if k in overrides
    }
    if bert_overrides:
        cfg["bert_config"] = _resolve_bert_config_path(cfg["bert_config"], bert_overrides)

    if target == "blip":
        tokenizer = init_blip_tokenizer()
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = "bert-base-uncased"

    if task == "vlr":
        model = RetrievalModel(
            config=cfg, text_encoder=text_encoder, tokenizer=tokenizer
        )
    else:
        model = VEModel(config=cfg, text_encoder=text_encoder, tokenizer=tokenizer)

    if target == "blip":
        sd = _co_attack_state_dict(state_dict, target)
    else:
        sd = _normalize_state_dict(state_dict, target)

    if "visual_encoder.pos_embed" in sd:
        sd["visual_encoder.pos_embed"] = interpolate_pos_embed(
            sd["visual_encoder.pos_embed"], model.visual_encoder
        )
    if "visual_encoder_m.pos_embed" in sd and hasattr(model, "visual_encoder_m"):
        sd["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
            sd["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )

    compatible = _filter_compatible(model, sd)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {list(missing)[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}...")

    model = model.to(device)
    print(f"Checkpoint loaded from {checkpoint_path} (target={target})")
    return model, tokenizer


def load_clip_model(checkpoint: str, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    hf_id = checkpoint or "openai/clip-vit-base-patch16"
    if hf_id.startswith("hf:"):
        hf_id = hf_id[3:]
    if Path(hf_id).is_dir():
        local_dir = Path(hf_id)
    else:
        root = Path(__file__).resolve().parent.parent
        local_dir = root / "checkpoints" / "vlr" / "clip" / hf_id.replace("/", "--")

    saved_endpoint = os.environ.get("HF_ENDPOINT")

    def _try_load(path_or_id, local_only=False):
        return CLIPModel.from_pretrained(path_or_id, local_files_only=local_only)

    def _try_processor(path_or_id, local_only=False):
        return CLIPProcessor.from_pretrained(path_or_id, local_files_only=local_only)

    model = None
    processor_source = hf_id
    if local_dir.is_dir():
        try:
            model = _try_load(str(local_dir), local_only=True)
            processor_source = str(local_dir)
            print(f"CLIP 从本地目录加载: {local_dir}")
        except OSError:
            pass

    if model is None:
        try:
            model = _try_load(hf_id, local_only=True)
            print(f"CLIP 从 HuggingFace 缓存加载: {hf_id}")
        except OSError:
            pass

    if model is None:
        endpoints = [
            "https://huggingface.co",
            "https://hf-mirror.com",
        ]
        if saved_endpoint and saved_endpoint not in endpoints:
            endpoints.append(saved_endpoint)
        last_err = None
        for endpoint in endpoints:
            os.environ["HF_ENDPOINT"] = endpoint
            try:
                model = _try_load(hf_id, local_only=False)
                print(f"CLIP 在线下载 ({endpoint}): {hf_id}")
                try:
                    local_dir.parent.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(local_dir))
                    _try_processor(hf_id, local_only=False).save_pretrained(str(local_dir))
                    processor_source = str(local_dir)
                    print(f"CLIP 已缓存到: {local_dir}")
                except OSError as exc:
                    print(f"CLIP 本地缓存写入失败（可忽略）: {exc}")
                break
            except OSError as exc:
                last_err = exc
        if model is None:
            if saved_endpoint is not None:
                os.environ["HF_ENDPOINT"] = saved_endpoint
            elif "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]
            raise OSError(
                f"无法加载 CLIP {hf_id}。可预下载到 {local_dir}：\n"
                f"  huggingface-cli download {hf_id} --local-dir {local_dir}\n"
                f"最后错误: {last_err}"
            )

    if saved_endpoint is not None:
        os.environ["HF_ENDPOINT"] = saved_endpoint

    processor = _try_processor(
        processor_source,
        local_only=Path(processor_source).is_dir(),
    )
    return model.to(device), processor
