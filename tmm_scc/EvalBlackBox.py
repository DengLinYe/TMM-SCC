import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import utils
from dataset.utils import pre_caption
from models.model_retrieval import ALBEF as TCL_Retrieval
from models.model_ve import ALBEF as TCL_VE
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

TEST_MODE = False
CLIP_HF_ID = "openai/clip-vit-base-patch16"

CONFIG_BY_TASK = {
    "vlr": "./tmm_scc/configs/Retrieval_flickr.yaml",
    "ve": "./tmm_scc/configs/ve_snli-ve.yaml",
}


def _resolve_project_path(path: str) -> str:
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    root = Path(__file__).resolve().parent.parent
    if path.startswith("./"):
        return str((root / path[2:]).resolve())
    return str((root / path).resolve())


class pair_dataset_blackbox(Dataset):
    """完整子集检索结构 + manifest 替换被攻击 (image, text) 对。"""

    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        manifest_file="",
        adv_image_root="",
        clean_eval=False,
        max_words=30,
    ):
        self.ann = json.load(open(ann_file, "r", encoding="utf-8"))
        if TEST_MODE:
            self.ann = self.ann[:5]
        self.transform = transform
        self.image_root = image_root
        self.adv_image_root = adv_image_root or ""
        self.clean_eval = clean_eval
        self.max_words = max_words
        self.adv_by_text = {}
        if manifest_file and Path(manifest_file).is_file() and not clean_eval:
            manifest = json.load(open(manifest_file, "r", encoding="utf-8"))
            for rec in manifest:
                self.adv_by_text[int(rec["text_id"])] = rec

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.image_ids = {}
        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            if "image_path" in ann:
                img_name = str(ann["image_path"])
            else:
                img_name = str(ann.get("image") or ann.get("image_id"))
                if not img_name.endswith((".jpg", ".png")):
                    img_name += ".jpg"
            captions = ann.get("caption", [ann.get("sentence", "")])
            if isinstance(captions, str):
                captions = [captions]
            for caption in captions:
                self.image.append(img_name)
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                if "image_id" in ann:
                    self.image_ids[txt_id] = str(ann["image_id"])
                else:
                    basename = os.path.splitext(os.path.basename(img_name))[0]
                    self.image_ids[txt_id] = basename
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        adv = self.adv_by_text.get(index)
        if adv and self.adv_image_root and not self.clean_eval:
            image_path = os.path.join(self.adv_image_root, adv["image_path"])
            caption = adv.get("adv_text") or self.text[index]
        else:
            image_path = os.path.join(self.image_root, self.image[index])
            caption = self.text[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, caption, index, self.image_ids[index]


class ve_dataset_blackbox(Dataset):
    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        manifest_file="",
        adv_image_root="",
        clean_eval=False,
        max_words=30,
    ):
        self.ann = json.load(open(ann_file, "r", encoding="utf-8"))
        if TEST_MODE:
            self.ann = self.ann[:5]
        self.transform = transform
        self.image_root = image_root
        self.adv_image_root = adv_image_root or ""
        self.clean_eval = clean_eval
        self.max_words = max_words
        self.label_map = {"entailment": 2, "neutral": 1, "contradiction": 0}
        self.adv_by_idx = {}
        if manifest_file and Path(manifest_file).is_file() and not clean_eval:
            manifest = json.load(open(manifest_file, "r", encoding="utf-8"))
            for rec in manifest:
                idx = int(rec.get("image_id", len(self.adv_by_idx)))
                self.adv_by_idx[idx] = rec

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        item = self.ann[index]
        adv = self.adv_by_idx.get(index)
        if adv and self.adv_image_root and not self.clean_eval:
            image_path = os.path.join(self.adv_image_root, adv["image_path"])
            text_content = adv.get("adv_text") or item.get("sentence", "")
        else:
            if "image_path" in item:
                img_name = str(item["image_path"])
            else:
                img_id = item.get("image") or item.get("image_id")
                img_name = str(img_id)
                if not img_name.endswith((".jpg", ".png")):
                    img_name += ".jpg"
            image_path = os.path.join(self.image_root, img_name)
            text_content = item.get("sentence") or item.get("caption", "")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        text = pre_caption(text_content, self.max_words)
        raw_label = item["label"]
        label = raw_label if isinstance(raw_label, int) else self.label_map[raw_label]
        return image, text, label


def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img):
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        gt_text_ids = img2txt[index]
        best_rank = 1e10
        for i in gt_text_ids:
            rank_pos = np.where(inds == i)[0][0]
            if rank_pos < best_rank:
                best_rank = rank_pos
        ranks[index] = best_rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    return {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "txt_r_mean": tr_mean,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
    }


def retrieval_score_itm(
    model, image_feats, image_embeds, text_feats, text_embeds, text_atts, config, device
):
    num_image = image_feats.shape[0]
    num_text = text_feats.shape[0]
    k_i2t = min(int(config["k_test"]), num_text)
    k_t2i = min(int(config["k_test"]), num_image)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Blackbox ITM rerank:"

    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    with torch.no_grad():
        for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
            topk_sim, topk_idx = sims.topk(k=k_i2t, dim=0)
            encoder_output = image_embeds[i].repeat(k_i2t, 1, 1).to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                device
            )
            output = model.text_encoder(
                encoder_embeds=text_embeds[topk_idx].to(device),
                attention_mask=text_atts[topk_idx].to(device),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
                mode="fusion",
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[i, topk_idx] = score

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
            topk_sim, topk_idx = sims.topk(k=k_t2i, dim=0)
            encoder_output = image_embeds[topk_idx].to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                device
            )
            output = model.text_encoder(
                encoder_embeds=text_embeds[i].repeat(k_t2i, 1, 1).to(device),
                attention_mask=text_atts[i].repeat(k_t2i, 1).to(device),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
                mode="fusion",
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def retrieval_score_blip_itm(
    model, image_feats, image_embeds, text_feats, text_ids, text_atts, config, device
):
    num_image = image_feats.shape[0]
    num_text = text_feats.shape[0]
    k_test = min(
        int(config.get("k_test_blip", config.get("k_test", 128))),
        num_text,
        num_image,
    )

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "BLIP ITM rerank:"

    sims_matrix = image_embeds @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    with torch.no_grad():
        for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            encoder_output = image_feats[i].unsqueeze(0).expand(k_test, -1, -1).to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                device
            )
            output = model.text_encoder(
                text_ids[topk_idx].to(device),
                attention_mask=text_atts[topk_idx].to(device),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
                mode="multi_modal",
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[i, topk_idx] = score + topk_sim.to(device)

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            encoder_output = image_feats[topk_idx].to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                device
            )
            output = model.text_encoder(
                text_ids[i].unsqueeze(0).expand(k_test, -1).to(device),
                attention_mask=text_atts[i].unsqueeze(0).expand(k_test, -1).to(device),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
                mode="multi_modal",
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score + topk_sim.to(device)

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def _effective_retrieval_texts(dataset):
    texts = list(dataset.text)
    if getattr(dataset, "clean_eval", False):
        return texts
    adv_by_text = getattr(dataset, "adv_by_text", None) or {}
    for tid, adv in adv_by_text.items():
        adv_text = adv.get("adv_text")
        if adv_text:
            texts[int(tid)] = pre_caption(adv_text, dataset.max_words)
    return texts


def evaluate_vlr_blip_itm(model, data_loader, tokenizer, device, config):
    model.eval()
    dataset = data_loader.dataset
    num_text = len(dataset.text)
    num_image = len(dataset.ann)
    max_length = 35
    enc_token_id = tokenizer.enc_token_id

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )

    texts = _effective_retrieval_texts(dataset)
    text_bs = 256
    text_feat_chunks = []
    text_id_chunks = []
    text_att_chunks = []
    for i in range(0, num_text, text_bs):
        batch = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_output = model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_feat = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
        text_feat_chunks.append(text_feat.cpu())
        text_id_chunks.append(text_input.input_ids.cpu())
        text_att_chunks.append(text_input.attention_mask.cpu())

    text_feats = torch.cat(text_feat_chunks, dim=0)
    text_ids = torch.cat(text_id_chunks, dim=0)
    text_atts = torch.cat(text_att_chunks, dim=0)
    text_ids[:, 0] = enc_token_id

    image_feats = torch.zeros(num_image, 577, 768)
    image_embeds = torch.zeros(num_image, config["embed_dim"])
    for images, _, text_ids_batch, _ in tqdm(data_loader, ascii=True, desc="blip encode"):
        images = normalize(images.to(device))
        with torch.no_grad():
            vit_out = model.visual_encoder(images)
            proj = F.normalize(model.vision_proj(vit_out[:, 0, :]), dim=-1)
        for j, tid in enumerate(text_ids_batch):
            img_idx = dataset.txt2img[int(tid)]
            image_feats[img_idx] = vit_out[j].cpu()
            image_embeds[img_idx] = proj[j].cpu()
        torch.cuda.empty_cache()

    return retrieval_score_blip_itm(
        model,
        image_feats,
        image_embeds,
        text_feats,
        text_ids,
        text_atts,
        config,
        device,
    )


def evaluate_vlr_albef_itm(model, data_loader, tokenizer, device, config):
    model.eval()
    dataset = data_loader.dataset
    num_text = len(dataset.text)
    num_image = len(dataset.ann)

    image_feats = torch.zeros(num_image, config["embed_dim"])
    image_embeds = torch.zeros(num_image, 577, 768)
    text_feats = torch.zeros(num_text, config["embed_dim"])
    text_embeds = torch.zeros(num_text, 30, 768)
    text_atts = torch.zeros(num_text, 30).long()

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )

    for images, texts, text_ids, _ in tqdm(data_loader, ascii=True, desc="encode"):
        images = normalize(images.to(device))
        text_ids_list = [int(text_ids[i]) for i in range(len(text_ids))]
        texts_input = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        img_indices = [dataset.txt2img[tid] for tid in text_ids_list]
        with torch.no_grad():
            output = model.inference(images, texts_input, use_embeds=False)
            for j, img_idx in enumerate(img_indices):
                tid = text_ids_list[j]
                image_feats[img_idx] = output["image_feat"][j].cpu()
                image_embeds[img_idx] = output["image_embed"][j].cpu()
                text_feats[tid] = output["text_feat"][j].cpu()
                text_embeds[tid] = output["text_embed"][j].cpu()
                text_atts[tid] = texts_input.attention_mask[j].cpu()
        torch.cuda.empty_cache()

    return retrieval_score_itm(
        model,
        image_feats,
        image_embeds,
        text_feats,
        text_embeds,
        text_atts,
        config,
        device,
    )


def _clip_feature_tensor(feat):
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    if hasattr(feat, "text_embeds") and feat.text_embeds is not None:
        return feat.text_embeds
    if hasattr(feat, "image_embeds") and feat.image_embeds is not None:
        return feat.image_embeds
    if hasattr(feat, "last_hidden_state"):
        return feat.last_hidden_state[:, 0, :]
    raise TypeError(f"无法从 {type(feat)} 提取 CLIP 特征向量")


def evaluate_vlr_clip(model, processor, data_loader, device):
    model.eval()
    dataset = data_loader.dataset
    num_text = len(dataset.text)
    num_image = len(dataset.ann)
    texts = dataset.text

    text_bs = 64
    text_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), text_bs):
            batch = texts[i : min(i + text_bs, len(texts))]
            inputs = processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = _clip_feature_tensor(
                model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
            )
            text_embeds.append(F.normalize(feats, dim=-1).cpu())
        text_embeds = torch.cat(text_embeds, dim=0)

        image_embeds = torch.zeros(num_image, text_embeds.shape[1])
        to_pil = transforms.ToPILImage()
        for images, _, text_ids, _ in tqdm(data_loader, ascii=True, desc="clip img"):
            for j in range(len(text_ids)):
                tid = int(text_ids[j])
                img_idx = dataset.txt2img[tid]
                pil = to_pil(images[j].clamp(0, 1))
                inputs = processor(images=pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                feat = _clip_feature_tensor(
                    model.get_image_features(pixel_values=inputs["pixel_values"])
                )
                image_embeds[img_idx] = F.normalize(feat, dim=-1).cpu().squeeze(0)

    sims_matrix = image_embeds @ text_embeds.t()
    return sims_matrix.numpy(), sims_matrix.t().numpy()


def evaluate_ve(model, data_loader, tokenizer, device):
    model.eval()
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for image, text, label in data_loader:
            image = normalize(image.to(device))
            label = label.to(device)
            text_input = tokenizer(
                list(text),
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            ).to(device)
            output = model(image, text_input, targets=label, train=False)
            _, pred = output.max(1)
            correct += (pred == label).sum().item()
            total += image.size(0)
    return 100.0 * correct / total


def load_albef_victim(task, config, checkpoint_path, device, target="tcl"):
    from victim_loader import load_victim_checkpoint

    return load_victim_checkpoint(
        task, config, checkpoint_path, device, target=target
    )


def run_single(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    yaml_parser = YAML(typ="safe")
    config_path = _resolve_project_path(args.config or CONFIG_BY_TASK[args.task])
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml_parser.load(f)

    args.annotation = _resolve_project_path(args.annotation)
    args.image_root = _resolve_project_path(args.image_root)
    if args.manifest:
        args.manifest = _resolve_project_path(args.manifest)
    if args.adv_image_root:
        args.adv_image_root = _resolve_project_path(args.adv_image_root)

    if not args.annotation:
        raise ValueError("需要 --annotation 指定子集标注文件")

    clean_eval = getattr(args, "clean_eval", False)

    if args.task == "vlr" and args.target == "clip":
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (config["image_res"], config["image_res"]),
                    interpolation=Image.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )

    manifest = "" if clean_eval else args.manifest
    adv_root = "" if clean_eval else args.adv_image_root

    if args.task == "vlr":
        test_dataset = pair_dataset_blackbox(
            args.annotation,
            test_transform,
            args.image_root,
            manifest_file=manifest,
            adv_image_root=adv_root,
            clean_eval=clean_eval,
        )
    else:
        test_dataset = ve_dataset_blackbox(
            args.annotation,
            test_transform,
            args.image_root,
            manifest_file=manifest,
            adv_image_root=adv_root,
            clean_eval=clean_eval,
        )

    drop_last = args.task == "vlr"
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size_test"],
        num_workers=0,
        drop_last=drop_last,
    )

    processor = None
    if args.task == "vlr" and args.target == "clip":
        from victim_loader import load_clip_model

        model, processor = load_clip_model(args.checkpoint, device)
        tokenizer = None
    else:
        if not args.checkpoint:
            raise ValueError(f"target={args.target} 需要 --checkpoint")
        ckpt_path = args.checkpoint
        if ckpt_path.startswith("hf:"):
            pass
        else:
            ckpt_path = _resolve_project_path(ckpt_path)
        model, tokenizer = load_albef_victim(
            args.task, config, ckpt_path, device, target=args.target
        )

    dump_path = getattr(args, "dump_metrics", "") or ""
    if args.task == "vlr":
        if args.target == "clip":
            score_i2t, score_t2i = evaluate_vlr_clip(
                model, processor, test_loader, device
            )
        elif args.target == "blip":
            score_i2t, score_t2i = evaluate_vlr_blip_itm(
                model, test_loader, tokenizer, device, config
            )
        else:
            score_i2t, score_t2i = evaluate_vlr_albef_itm(
                model, test_loader, tokenizer, device, config
            )
        raw = itm_eval(
            score_i2t,
            score_t2i,
            test_loader.dataset.img2txt,
            test_loader.dataset.txt2img,
        )
    else:
        raw = {"accuracy": evaluate_ve(model, test_loader, tokenizer, device)}

    if dump_path:
        dump_file = Path(_resolve_project_path(dump_path))
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        dump_file.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return raw

    if not clean_eval:
        utils.write_blackbox_run_log(args.log, args, raw)

    print(json.dumps(raw, ensure_ascii=False, indent=2))
    return raw


def main():
    parser = argparse.ArgumentParser(description="黑盒迁移攻击评测")
    parser.add_argument("--task", choices=["vlr", "ve"], required=True)
    parser.add_argument("--target", default="tcl")
    parser.add_argument("--method", default="tmm")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--adv-image-root", default="")
    parser.add_argument("--annotation", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default=None)
    parser.add_argument("--subset", default="main_1k")
    parser.add_argument("--log", default="./outputs/run.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--clean-eval", action="store_true")
    parser.add_argument("--dump-metrics", default="")
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    run_single(args)


if __name__ == "__main__":
    main()
