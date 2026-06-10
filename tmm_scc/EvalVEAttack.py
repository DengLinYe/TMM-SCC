import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import BertForMaskedLM

import utils
from attack import MultiModalAttacker
from dataset.caption_dataset_ve import ve_dataset_attack
from models.model_ve import ALBEF
from models.tokenization_bert import BertTokenizer


class ModelVE:
    def __init__(self, config, text_encoder_name, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.model = ALBEF(
            config=config, text_encoder=text_encoder_name, tokenizer=tokenizer
        )
        self.ref_model = BertForMaskedLM.from_pretrained(text_encoder_name)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        ret = self.model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", ret.missing_keys)
        print("Checkpoint loaded from %s" % checkpoint_path)

    def to_device(self, device):
        self.model = self.model.to(device)
        self.ref_model = self.ref_model.to(device)

    def inference(self, images, texts_input, use_embeds):
        return self.model.inference(images, texts_input, use_embeds=use_embeds)


class EvaluationVE:
    def __init__(self, model, ref_model, data_loader, tokenizer, device, config):
        self.model = model
        self.ref_model = ref_model
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

    def ve_eval(self, args):
        self.model.eval()
        self.ref_model.eval()

        clean_mode = args.adv == 0 or getattr(args, "clean_eval", False)
        label = "clean" if clean_mode else "adv"
        print(f"Computing features for VE evaluation ({label})...")
        start_time = time.time()

        total_sim = 0.0
        sim_count = 0

        images_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        multi_attacker = None
        sim_model = None
        if not clean_mode:
            sim_model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
            device = next(self.model.parameters()).device
            if args.text_method == "tmm":
                from attack.textAttack import TextAttacker

                print(">>> 加载原版 TMM 文本攻击器...")
                text_attacker = TextAttacker(
                    self.ref_model,
                    self.tokenizer,
                    device,
                    cls=args.cls,
                )
            elif args.text_method == "scc":
                from attack.textAttackSCC import TextAttackerSCC

                print(f">>> 加载 TMM-SCC 文本攻击器 (阈值: {args.sim_threshold})...")
                text_attacker = TextAttackerSCC(
                    tokenizer=self.tokenizer,
                    device=device,
                    cls=args.cls,
                    sim_threshold=args.sim_threshold,
                )
            else:
                raise ValueError(f"不支持的文本攻击方法: {args.text_method}")

            multi_attacker = MultiModalAttacker(
                net=self.model,
                text_attacker=text_attacker,
                tokenizer=self.tokenizer,
                args=args,
                cls=args.cls,
            )

        correct = 0
        total = 0

        adv_save_dir = os.path.join(args.save_dir, "adv_samples_ve")
        adv_records = []
        if not clean_mode:
            os.makedirs(adv_save_dir, exist_ok=True)

        import torchvision

        print("Forward VE")
        for step, (images, texts, labels) in enumerate(
            tqdm(self.data_loader, ascii=True)
        ):
            images = images.to(self.device)
            labels = labels.to(self.device)
            orig_texts = list(texts)

            if args.adv != 0 and multi_attacker is not None:
                images, texts = multi_attacker.run_transfer_attack(
                    images, texts, args, num_iters=self.config["num_iters"]
                )

            if not clean_mode:
                orig_embs = sim_model.encode(orig_texts, convert_to_tensor=True)
                adv_embs = sim_model.encode(texts, convert_to_tensor=True)
                batch_sims = F.cosine_similarity(orig_embs, adv_embs)

                total_sim += batch_sims.sum().item()
                sim_count += len(texts)

                for i in range(images.size(0)):
                    global_idx = step * self.config["batch_size_test"] + i
                    img_filename = f"adv_{global_idx}.png"
                    img_path = os.path.join(adv_save_dir, img_filename)
                    torchvision.utils.save_image(images[i], img_path)

                    adv_records.append(
                        {
                            "image_id": global_idx,
                            "image_path": img_filename,
                            "orig_text": orig_texts[i],
                            "adv_text": texts[i],
                            "label": labels[i].item(),
                            "sim_score": round(batch_sims[i].item(), 4),
                        }
                    )

            texts_input = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                images = images_normalize(images)
                prediction = self.model(
                    images, texts_input, targets=labels, train=False
                )

                _, predicted_class = prediction.max(1)
                total += labels.size(0)
                correct += predicted_class.eq(labels).sum().item()

            torch.cuda.empty_cache()

        if not clean_mode:
            with open(
                os.path.join(args.save_dir, "ve_adv_manifest.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(adv_records, f, indent=4, ensure_ascii=False)

        accuracy = 100.0 * correct / total
        avg_sim = total_sim / sim_count if sim_count > 0 else 0.0

        print(f"VE Task Accuracy after Attack: {accuracy:.2f}%")
        if not clean_mode:
            print(f"Average Semantic Similarity: {avg_sim:.4f}")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Evaluation time {}".format(total_time_str))

        return {"accuracy": accuracy, "avg_sim": avg_sim}


def main(args, config):
    print("starting VE attack evaluation...")
    device = args.gpu[0]

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating VE dataset")
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
        ]
    )

    test_dataset = ve_dataset_attack(
        config["test_file"], test_transform, config["image_root"]
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size_test"], num_workers=0
    )

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    model_handler = ModelVE(config, args.text_encoder, tokenizer)
    model_handler.load_checkpoint(args.checkpoint)
    model_handler.to_device(device)

    eval_handler = EvaluationVE(
        model_handler.model,
        model_handler.ref_model,
        test_loader,
        tokenizer,
        device,
        config,
    )

    result = eval_handler.ve_eval(args)
    print(result)

    dump_path = getattr(args, "dump_metrics", "") or ""
    if dump_path:
        dump_file = Path(dump_path)
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        dump_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if getattr(args, "clean_eval", False):
        return

    subset = getattr(args, "subset", "") or ""
    model = getattr(args, "model", "albef") or "albef"
    metrics = utils.format_result_metrics("ve", result, subset=subset, model=model)
    log_stats = {
        **metrics,
        "text_method": args.text_method,
        "sim_threshold": args.sim_threshold,
        "eval type": args.adv,
        "cls": args.cls,
        "eps": config.get("epsilon", 12),
        "iters": config.get("num_iters", 10),
        "alpha": config.get("alpha", 0.4),
        "intervals": getattr(args, "intervals", 5),
        "num_steps": getattr(args, "num_steps", 10),
        "kernel_size": getattr(args, "kernel_size", 5),
        "momentum": getattr(args, "momentum", 1.0),
        "mode": getattr(args, "mode", "nearest"),
    }
    print(log_stats)
    with open(os.path.join(args.output_dir, args.log_name), "w", encoding="utf-8") as f:
        json.dump(log_stats, f, ensure_ascii=False, indent=2)

    if getattr(args, "run_log", ""):
        utils.write_attack_run_log(
            args.run_log,
            args,
            config,
            "ve",
            result,
            dataset_meta={"num_samples": len(test_dataset)},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="snli-ve")
    parser.add_argument("--config", default="./tmm_scc/configs/ve_snli-ve.yaml")
    parser.add_argument("--output_dir", default="./outputs/whitebox/ve/albef/tmm/main_1k")
    parser.add_argument("--checkpoint", default="./checkpoints/ve/albef_ve_snli_ve.pth")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--adv", default=1, type=int)
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--lr", default=3e-4)
    parser.add_argument("--intervals", default=5, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--momentum", default=1, type=float)
    parser.add_argument("--mode", type=str, default="nearest")
    parser.add_argument("--epsilon", default=12, type=float)
    parser.add_argument("--epsilon_per", default=0.4, type=float)
    parser.add_argument("--log_name", default="ve_attack_log.txt")
    parser.add_argument("--save_json_name", default="")
    parser.add_argument("--config_name", default="config.yaml")
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--run_log", default="./outputs/run.json")
    parser.add_argument("--subset", default="")
    parser.add_argument("--model", default="albef")
    parser.add_argument("--att_mask", default=0.1, type=float)
    parser.add_argument("--clean_eval", action="store_true")
    parser.add_argument("--dump_metrics", default="")

    parser.add_argument(
        "--text_method",
        type=str,
        default="tmm",
        choices=["tmm", "scc"],
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.65,
    )

    args = parser.parse_args()

    yaml_parser = yaml.YAML(typ="safe")
    config = yaml_parser.load(open(args.config, "r", encoding="utf-8"))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml = yaml.YAML(typ="unsafe", pure=True)
    yaml.dump(config, open(os.path.join(args.output_dir, args.config_name), "w"))

    main(args, config)
