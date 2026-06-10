import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from transformers import BertForMaskedLM

from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import ve_dataset
from PIL import Image
from torchvision import transforms
from attack.bert_attack import BertAttack
from attack.imageAttack import ImageAttacker
from attack.multimodalAttack import MultiModalAttacker

def evaluate(model, ref_model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    ref_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image_attacker = ImageAttacker(config['epsilon'] / 255., preprocess=images_normalize, bounding=(0, 1), cls=args.cls)
    text_attacker = BertAttack(ref_model, tokenizer, cls=args.cls)
    multi_attacker = MultiModalAttacker(model, image_attacker, text_attacker, tokenizer, cls=args.cls)

    save_dir = getattr(args, "save_dir", "") or getattr(args, "output_dir", "")
    export_adv = args.adv != 0 and bool(save_dir)
    adv_records = []
    adv_save_dir = ""
    if export_adv:
        from baselines.export_adv import append_ve_records, ve_adv_dir, write_ve_manifest

        adv_save_dir = ve_adv_dir(save_dir)

    for step, (images, text, targets) in enumerate(data_loader):
        images, targets = images.to(device), targets.to(device)
        orig_texts = list(text)

        if args.adv != 0:
            images, text = multi_attacker.run_before_fusion(images, text, adv=args.adv, num_iters=config['num_iters'],
                                                            alpha=args.alpha)

        if export_adv:
            append_ve_records(
                adv_records,
                adv_save_dir,
                images,
                orig_texts,
                list(text),
                targets,
                step,
                config["batch_size_test"],
            )

        images = images_normalize(images)
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        with torch.no_grad():
            prediction = model(images, text_inputs, targets=targets, train=False)

        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item() * 100.0, n=images.size(0))

    if export_adv:
        write_ve_manifest(save_dir, adv_records)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    device = args.gpu[0]

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    datasets = ve_dataset(config['test_file'], test_transform, config['image_root'])
    test_loader = DataLoader(datasets, batch_size=config['batch_size_test'], num_workers=0)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint

    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped


    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % args.checkpoint)
    #print(msg)

    model = model.to(device)
    ref_model = ref_model.to(device)


    print("Start evaluating")
    start_time = time.time()

    test_stats = evaluate(model, ref_model, test_loader, tokenizer, device, config)

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'eval type': args.adv, 'cls': args.cls,
                 'eps': config['epsilon'], 'iters':config['num_iters'], 'alpha': args.alpha}

    with open(os.path.join(args.output_dir, "log.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VE.yaml')
    parser.add_argument('--output_dir', default='output/VE')
    parser.add_argument('--checkpoint', default='checkpoints/VE.pth')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--gpu', type=int, nargs='+',  default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=0, type=int)
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--save_dir', default='')
    args = parser.parse_args()
    if not args.save_dir:
        args.save_dir = args.output_dir

    yaml_parser = yaml.YAML(typ="safe")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml_parser.load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml_parser.dump(config, f)

    main(args, config)
