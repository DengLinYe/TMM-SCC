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
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import BertForMaskedLM

import utils
from attack import MultiModalAttacker, TextAttacker
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
        self.model.load_state_dict(state_dict, strict=False)
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

        print("Computing features for VE evaluation adv...")
        start_time = time.time()

        images_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        text_attacker = TextAttacker(
            self.ref_model,
            self.tokenizer,
            next(self.model.parameters()).device,
            cls=args.cls,
        )
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
        os.makedirs(adv_save_dir, exist_ok=True)
        adv_records = []

        import torchvision

        print("Forward VE")
        for step, (images, texts, labels) in enumerate(
            tqdm(self.data_loader, ascii=True)
        ):
            if step >= 10:
                print("\n>>> 快速验证：已完成 10 个 Batch，提前结束攻击循环 <<<")
                torch.cuda.empty_cache()
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            if args.adv != 0:
                images, texts = multi_attacker.run_transfer_attack(
                    images, texts, args, num_iters=self.config["num_iters"]
                )

            for i in range(images.size(0)):
                global_idx = step * self.config["batch_size_test"] + i
                img_filename = f"adv_{global_idx}.png"
                img_path = os.path.join(adv_save_dir, img_filename)
                torchvision.utils.save_image(images[i], img_path)

                adv_records.append(
                    {
                        "image_id": global_idx,
                        "image_path": img_filename,
                        "adv_text": texts[i],
                        "label": labels[i].item(),
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

        with open(
            os.path.join(args.save_dir, "ve_adv_manifest.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(adv_records, f, indent=4, ensure_ascii=False)

        accuracy = 100.0 * correct / total
        print(f"VE Task Accuracy after Attack: {accuracy:.2f}%")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Evaluation time {}".format(total_time_str))

        return {"accuracy": accuracy}


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

    log_stats = {
        **{f"test_{k}": v for k, v in result.items()},
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
    with open(os.path.join(args.output_dir, args.log_name), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="snli-ve")
    parser.add_argument("--config", default="./configs/ve_snli-ve.yaml")
    parser.add_argument("--output_dir", default="./output/ve/snli-ve")
    parser.add_argument("--checkpoint", default="./checkpoints/ALBEF/ve.pth")
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
    parser.add_argument("--att_mask", default=0.1, type=int)

    args = parser.parse_args()

    yaml_parser = yaml.YAML(typ="safe")
    config = yaml_parser.load(open(args.config, "r", encoding="utf-8"))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml = yaml.YAML(typ="unsafe", pure=True)
    yaml.dump(config, open(os.path.join(args.output_dir, args.config_name), "w"))

    main(args, config)
