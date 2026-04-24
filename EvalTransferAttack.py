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
from dataset import pair_dataset_attack
from models.model_retrieval import ALBEF
from models.tokenization_bert import BertTokenizer


class Model:
    def __init__(self, config, text_encoder_name, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.model = ALBEF(
            config=config, text_encoder=text_encoder_name, tokenizer=tokenizer
        )
        self.ref_model = BertForMaskedLM.from_pretrained(text_encoder_name)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded from %s" % checkpoint_path)

    def to_device(self, device):
        self.model = self.model.to(device)
        self.ref_model = self.ref_model.to(device)

    def inference(self, images, texts_input, use_embeds):
        return self.model.inference(images, texts_input, use_embeds=use_embeds)


class Evaluation:
    def __init__(self, model, ref_model, data_loader, tokenizer, device, config):
        self.model = model
        self.ref_model = ref_model
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

    def retrieval_eval(self):
        self.model.eval()
        self.ref_model.eval()

        print("Computing features for evaluation adv...")
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

        print("Prepare memory")
        num_text = len(self.data_loader.dataset.text)
        num_image = len(self.data_loader.dataset.ann)

        image_feats = torch.zeros(num_image, config["embed_dim"])
        image_embeds = torch.zeros(num_image, 577, 768)

        text_feats = torch.zeros(num_text, config["embed_dim"])
        text_embeds = torch.zeros(num_text, 30, 768)
        text_atts = torch.zeros(num_text, 30).long()
        adv_example = []
        print("Forward")
        for step, (images, texts, texts_ids, _) in enumerate(
            tqdm(self.data_loader, ascii=True)
        ):
            if step >= 10:
                print("\n>>> 快速验证：已完成 10 个 Batch，提前结束攻击循环 <<<")
                torch.cuda.empty_cache()
                break

            images = images.to(self.device)
            if args.adv != 0:
                images, texts = multi_attacker.run_transfer_attack(
                    images, texts, args, num_iters=config["num_iters"]
                )

            texts_input = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            ).to(self.device)
            images_ids = [self.data_loader.dataset.txt2img[i.item()] for i in texts_ids]
            with torch.no_grad():
                images = images_normalize(images)
                output = self.model.inference(images, texts_input, use_embeds=False)
                image_feats[images_ids] = output["image_feat"].cpu().detach()
                image_embeds[images_ids] = output["image_embed"].cpu().detach()
                text_feats[texts_ids] = output["text_feat"].cpu().detach()
                text_embeds[texts_ids] = output["text_embed"].cpu().detach()
                text_atts[texts_ids] = texts_input.attention_mask.cpu().detach()

            torch.cuda.empty_cache()

        with open(args.save_json_name, "w", encoding="utf8") as f:
            json.dump(adv_example, f, ensure_ascii=False, indent=2)

        score_matrix_i2t, score_matrix_t2i = self.retrieval_score(
            image_feats,
            image_embeds,
            text_feats,
            text_embeds,
            text_atts,
            num_image,
            num_text,
        )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Evaluation time {}".format(total_time_str))

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    def retrieval_score(
        self,
        image_feats,
        image_embeds,
        text_feats,
        text_embeds,
        text_atts,
        num_image,
        num_text,
    ):
        self.device = next(self.model.parameters()).device

        # self.config["k_test"] = 16  # 128

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Evaluation Direction Similarity With Bert Attack:"

        sims_matrix = image_feats @ text_feats.t()
        score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(self.device)

        with torch.no_grad():
            for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
                topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)

                encoder_output = (
                    image_embeds[i].repeat(config["k_test"], 1, 1).to(self.device)
                )
                encoder_att = torch.ones(
                    encoder_output.size()[:-1], dtype=torch.long
                ).to(self.device)
                output = self.model.text_encoder(
                    encoder_embeds=text_embeds[topk_idx].to(self.device),
                    attention_mask=text_atts[topk_idx].to(self.device),
                    encoder_hidden_states=encoder_output,
                    encoder_attention_mask=encoder_att,
                    return_dict=True,
                    mode="fusion",
                )
                score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                score_matrix_i2t[i, topk_idx] = score

            sims_matrix = sims_matrix.t()
            score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(self.device)

            for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
                topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
                encoder_output = image_embeds[topk_idx].to(self.device)
                encoder_att = torch.ones(
                    encoder_output.size()[:-1], dtype=torch.long
                ).to(self.device)
                output = self.model.text_encoder(
                    encoder_embeds=text_embeds[i]
                    .repeat(config["k_test"], 1, 1)
                    .to(self.device),
                    attention_mask=text_atts[i]
                    .repeat(config["k_test"], 1)
                    .to(self.device),
                    encoder_hidden_states=encoder_output,
                    encoder_attention_mask=encoder_att,
                    return_dict=True,
                    mode="fusion",
                )
                score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                score_matrix_t2i[i, topk_idx] = score

        return score_matrix_i2t, score_matrix_t2i

    def itm_eval(self, scores_i2t, scores_t2i, img2txt, txt2img):
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == img2txt[index][0])[0][0]
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


def main(args, config):
    print("starting...")
    device = args.gpu[0]

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
        ]
    )
    test_dataset = pair_dataset_attack(
        config["test_file"], test_transform, config["image_root"], args
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size_test"], num_workers=0
    )

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    model_handler = Model(config, args.text_encoder, tokenizer)
    model_handler.load_checkpoint(args.checkpoint)
    model_handler.to_device(device)

    eval_handler = Evaluation(
        model_handler.model,
        model_handler.ref_model,
        test_loader,
        tokenizer,
        device,
        config,
    )
    score_i2t, score_t2i = eval_handler.retrieval_eval()
    result = eval_handler.itm_eval(
        score_i2t, score_t2i, test_dataset.img2txt, test_dataset.txt2img
    )
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
    print(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="flickr")
    parser.add_argument("--config", default="./configs/Retrieval_flickr.yaml")
    parser.add_argument("--output_dir", default="./output/retrieval/flickr")
    parser.add_argument("--checkpoint", default="./checkpoints/ALBEF/flickr30k.pth")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--adv", default=1, type=int, help="0=clean, 1=adv")
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--lr", default=3e-4)
    parser.add_argument("--intervals", default=5, type=int, help="number of intervals")
    parser.add_argument(
        "--kernel_size", default=5, type=int, help="kernel size of gaussian filter"
    )
    parser.add_argument(
        "--momentum", default=1, type=float, help="momentum, (default: 1.0)"
    )
    parser.add_argument("--mode", type=str, default="nearest")
    parser.add_argument(
        "--epsilon", default=12, type=float, help="perturbation, (default: 16)"
    )
    parser.add_argument("--epsilon_per", default=0.4, type=float)
    parser.add_argument("--log_name", default="")
    parser.add_argument("--save_json_name", default="")
    parser.add_argument("--config_name", default="")
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--att_mask", default=0.1, type=int)

    args = parser.parse_args()

    yaml_parser = yaml.YAML(typ="safe")
    config = yaml_parser.load(open(args.config, "r"))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml = yaml.YAML(typ="unsafe", pure=True)
    yaml.dump(config, open(os.path.join(args.output_dir, args.config_name), "w"))

    main(args, config)
