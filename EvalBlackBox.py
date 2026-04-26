import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.utils import pre_caption
from models.model_retrieval import ALBEF as TCL_Retrieval
from models.model_ve import ALBEF as TCL_VE
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class pair_dataset_attack_robust(Dataset):
    def __init__(self, ann_file, transform, image_root, args, max_words=30):
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.image_ids = {}
        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            img_name = (
                ann.get("image_path") or ann.get("image") or str(ann.get("image_id"))
            )
            if not os.path.splitext(img_name)[1]:
                img_name += ".jpg"

            captions = ann.get("caption") or [
                ann.get("adv_text") or ann.get("sentence", "")
            ]
            if isinstance(captions, str):
                captions = [captions]

            for j, caption in enumerate(captions):
                self.image.append(img_name)
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i

                raw_id = str(ann.get("image_id", i))
                if "/" in raw_id:
                    self.image_ids[txt_id] = raw_id.split("/")[-1].split(".")[0]
                else:
                    self.image_ids[txt_id] = raw_id.split(".")[0]

                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, self.text[index], index, self.image_ids[index]


class ve_dataset_attack_robust(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        item = self.ann[index]
        img_name = (
            item.get("image_path") or item.get("image") or str(item.get("image_id"))
        )
        if not os.path.splitext(img_name)[1]:
            img_name += ".jpg"
        image_path = os.path.join(self.image_root, img_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        text_content = (
            item.get("adv_text") or item.get("sentence") or item.get("caption", "")
        )
        text = pre_caption(text_content, self.max_words)
        raw_label = item["label"]
        label = raw_label if isinstance(raw_label, int) else self.label_map[raw_label]
        return image, text, label


try:
    from EvalTransferAttack import itm_eval
except ImportError:

    def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img):
        pass


def evaluate_vlr(model, data_loader, tokenizer, device):
    model.eval()
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    with torch.no_grad():
        for i in range(0, num_text, text_bs):
            text = texts[i : min(i + text_bs, num_text)]
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            ).to(device)
            text_output = model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                mode="text",
            )
            text_embed = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
            text_embeds.append(text_embed)
        text_embeds = torch.cat(text_embeds, dim=0)
        image_embeds = []
        seen_images = set()
        for image, _, index, image_id in data_loader:
            for b_img, b_id in zip(image, image_id):
                if b_id not in seen_images:
                    seen_images.add(b_id)
                    b_img = b_img.unsqueeze(0).to(device)
                    image_feat = model.visual_encoder(b_img)
                    image_embed = F.normalize(
                        model.visual_proj(image_feat[:, 0, :]), dim=-1
                    )
                    image_embeds.append(image_embed)
        image_embeds = torch.cat(image_embeds, dim=0)
        sims_matrix = image_embeds @ text_embeds.t()
    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()


def evaluate_ve(model, data_loader, tokenizer, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, text, label in data_loader:
            image = image.to(device)
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
    return correct / total


def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    if args.task == "vlr":
        test_dataset = pair_dataset_attack_robust(
            args.adv_json, test_transform, args.adv_image_root, args
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size_test"], num_workers=4
        )
        model = TCL_Retrieval(
            config=config, text_encoder=args.text_encoder, tokenizer=tokenizer
        )
    elif args.task == "ve":
        test_dataset = ve_dataset_attack_robust(
            args.adv_json, test_transform, args.adv_image_root
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size_test"], num_workers=4
        )
        model = TCL_VE(
            config=config, text_encoder=args.text_encoder, tokenizer=tokenizer
        )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    start_eval_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {}

    if args.task == "vlr":
        score_i2t, score_t2i = evaluate_vlr(model, test_loader, model.tokenizer, device)
        results = itm_eval(
            score_i2t,
            score_t2i,
            test_loader.dataset.img2txt,
            test_loader.dataset.txt2img,
        )
    elif args.task == "ve":
        acc = evaluate_ve(model, test_loader, model.tokenizer, device)
        results = {"accuracy": acc}

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_path = os.path.join(args.log_dir, "blackbox_results_log.json")
    log_data = {
        "timestamp": start_eval_time,
        "task": args.task,
        "adv_data": {"adv_json": args.adv_json, "adv_image_root": args.adv_image_root},
        "checkpoint": args.checkpoint,
        "results": results,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Retrieval_flickr.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--log_dir", default="./results_log")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--task", default="vlr", choices=["vlr", "ve"])
    parser.add_argument("--dataset", default="flickr", choices=["flickr"])
    parser.add_argument("--adv_json", required=True, type=str)
    parser.add_argument("--adv_image_root", required=True, type=str)
    args = parser.parse_args()
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f)
    main(args, config)
