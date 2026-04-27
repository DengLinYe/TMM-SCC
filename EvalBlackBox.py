import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.utils import pre_caption
from models.model_retrieval import ALBEF as TCL_Retrieval
from models.model_ve import ALBEF as TCL_VE
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 测试模式：仅使用前5条数据进行快速验证，正式评测时请设置为 False
TEST_MODE = False


class pair_dataset_attack_robust(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, "r"))

        if TEST_MODE:
            self.ann = self.ann[:5]

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

        if TEST_MODE:
            self.ann = self.ann[:5]

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

    import numpy as np


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

    eval_result = {
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
    return eval_result


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
                        model.vision_proj(image_feat[:, 0, :]), dim=-1
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


def main():
    device = torch.device("cuda")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    log_dir = "./output/result/results_log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "blackbox_results_log.json")

    yaml_parser = YAML(typ="safe")

    tasks = [
        {
            "task_name": "VE",
            "method": "SCC",
            "config": "./configs/ve_snli-ve.yaml",
            "checkpoint": "./checkpoints/tcl_ve_snli_ve.pth",
            "adv_json": "./output/VE_SCC/final/ve_adv_manifest.json",
            "adv_image_root": "./output/VE_SCC/final/adv_samples_ve",
        },
        {
            "task_name": "VE",
            "method": "TMM",
            "config": "./configs/ve_snli-ve.yaml",
            "checkpoint": "./checkpoints/tcl_ve_snli_ve.pth",
            "adv_json": "./output/VE_TMM/final/ve_adv_manifest.json",
            "adv_image_root": "./output/VE_TMM/final/adv_samples_ve",
        },
        {
            "task_name": "VLR",
            "method": "SCC",
            "config": "./configs/Retrieval_flickr.yaml",
            "checkpoint": "./checkpoints/tcl_retrieval_flickr.pth",
            "adv_json": "./output/VLR_SCC/ve_adv_manifest.json",
            "adv_image_root": "./output/VLR_SCC/adv_samples_retrieval",
        },
        {
            "task_name": "VLR",
            "method": "TMM",
            "config": "./configs/Retrieval_flickr.yaml",
            "checkpoint": "./checkpoints/tcl_retrieval_flickr.pth",
            "adv_json": "./output/VLR_TMM/ve_adv_manifest.json",
            "adv_image_root": "./output/VLR_TMM/adv_samples_retrieval",
        },
    ]

    for t in tasks:
        print(f"\n{'=' * 50}")
        print(f">>> 开始执行黑盒评估: 任务={t['task_name']}, 方法={t['method']}")
        print(f"{'=' * 50}")

        with open(t["config"], "r", encoding="utf-8") as f:
            config = yaml_parser.load(f)

        test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (config["image_res"], config["image_res"]),
                    interpolation=Image.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

        if t["task_name"] == "VLR":
            test_dataset = pair_dataset_attack_robust(
                t["adv_json"], test_transform, t["adv_image_root"]
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config["batch_size_test"], num_workers=4
            )
            model = TCL_Retrieval(
                config=config, text_encoder="bert-base-uncased", tokenizer=tokenizer
            )
        else:
            test_dataset = ve_dataset_attack_robust(
                t["adv_json"], test_transform, t["adv_image_root"]
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config["batch_size_test"], num_workers=4
            )
            model = TCL_VE(
                config=config, text_encoder="bert-base-uncased", tokenizer=tokenizer
            )

        checkpoint = torch.load(t["checkpoint"], map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

        start_eval_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = {}

        if t["task_name"] == "VLR":
            score_i2t, score_t2i = evaluate_vlr(model, test_loader, tokenizer, device)
            results = itm_eval(
                score_i2t,
                score_t2i,
                test_loader.dataset.img2txt,
                test_loader.dataset.txt2img,
            )
        else:
            acc = evaluate_ve(model, test_loader, tokenizer, device)
            results = {"accuracy": acc}

        log_data = {
            "timestamp": start_eval_time,
            "task": t["task_name"],
            "method": t["method"],
            "target_model": "TCL",
            "adv_data": {
                "adv_json": t["adv_json"],
                "adv_image_root": t["adv_image_root"],
            },
            "checkpoint": t["checkpoint"],
            "results": results,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

        print(f">>> {t['task_name']} ({t['method']}) 评估完成，结果已写入 JSON。")
        print(results)

        torch.cuda.empty_cache()
        time.sleep(30)

    print("\n>>> 所有黑盒评测任务执行完毕！")


if __name__ == "__main__":
    main()
