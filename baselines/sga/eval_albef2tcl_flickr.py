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

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils

from attacker import SGAttacker, ImageAttacker, TextAttacker

from dataset import paired_dataset

def retrieval_eval(
    model,
    ref_model,
    t_model,
    t_ref_model,
    t_test_transform,
    data_loader,
    tokenizer,
    t_tokenizer,
    device,
    config,
    scales=None,
    save_dir="",
):
    # test
    model.float()
    model.eval()
    ref_model.eval()
    t_model.float()
    t_model.eval()
    if t_ref_model is not ref_model:
        t_ref_model.eval()

    same_target = model is t_model

    print('Computing features for evaluation adv...')

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_attacker = ImageAttacker(images_normalize, eps=2/255, steps=10, step_size=0.5/255)
    txt_attacker = TextAttacker(ref_model, tokenizer, cls=False, max_length=30, number_perturbation=1,
                                topk=10, threshold_pred_score=0.3)
    attacker = SGAttacker(model, img_attacker, txt_attacker)

    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    s_image_feats = torch.zeros(num_image, config['embed_dim'])
    s_image_embeds = torch.zeros(num_image, 577, 768)
    s_text_feats = torch.zeros(num_text, config['embed_dim'])
    s_text_embeds = torch.zeros(num_text, 30, 768)
    s_text_atts = torch.zeros(num_text, 30).long()

    t_image_feats = torch.zeros(num_image, config['embed_dim'])
    t_image_embeds = torch.zeros(num_image, 577, 768)
    t_text_feats = torch.zeros(num_text, config['embed_dim'])
    t_text_embeds = torch.zeros(num_text, 30, 768)
    t_text_atts = torch.zeros(num_text, 30).long()

    export_adv = bool(save_dir)
    adv_records = []
    adv_save_dir = ""
    pair_offset = 0
    if export_adv:
        from baselines.export_adv import append_sga_vlr_batch, vlr_adv_dir, write_vlr_manifest

        adv_save_dir = vlr_adv_dir(save_dir)

    if scales is not None:
        print(scales)
    else:
        scales = None

    print('Forward')
    for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(data_loader):
        print(f'--------------------> batch:{batch_idx}/{len(data_loader)}')
        texts_ids = []
        txt2img = []
        texts = []
        for i in range(len(texts_group)):
            texts += texts_group[i]
            texts_ids += text_ids_groups[i]
            txt2img += [i] * len(text_ids_groups[i])

        images = images.to(device)
        orig_texts = list(texts)
        adv_images, adv_texts = attacker.attack(
            images, texts, txt2img, device=device, max_lemgth=30, scales=scales
        )
        torch.cuda.empty_cache()

        if export_adv:
            append_sga_vlr_batch(
                adv_records,
                adv_save_dir,
                adv_images,
                orig_texts,
                list(adv_texts),
                texts_ids,
                txt2img,
                pair_offset,
            )
            pair_offset += len(adv_texts)

        with torch.no_grad():
            s_adv_images_norm = images_normalize(adv_images)
            adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, 
                                        return_tensors="pt").to(device)            
            s_output_img = model.inference_image(s_adv_images_norm)
            s_output_txt = model.inference_text(adv_texts_input)
            s_image_feats[images_ids] = s_output_img['image_feat'].cpu().detach()
            s_image_embeds[images_ids] = s_output_img['image_embed'].cpu().detach()
            s_text_feats[texts_ids] = s_output_txt['text_feat'].cpu().detach()
            s_text_embeds[texts_ids] = s_output_txt['text_embed'].cpu().detach()
            s_text_atts[texts_ids] = adv_texts_input.attention_mask.cpu().detach()

            if same_target:
                t_image_feats[images_ids] = s_image_feats[images_ids]
                t_image_embeds[images_ids] = s_image_embeds[images_ids]
                t_text_feats[texts_ids] = s_text_feats[texts_ids]
                t_text_embeds[texts_ids] = s_text_embeds[texts_ids]
                t_text_atts[texts_ids] = s_text_atts[texts_ids]
            else:
                t_adv_img_list = []
                for itm in adv_images:
                    t_adv_img_list.append(t_test_transform(itm))
                t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)

                t_adv_images_norm = images_normalize(t_adv_imgs)
                adv_texts_input = tokenizer(
                    adv_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=30,
                    return_tensors="pt",
                ).to(device)
                t_output_img = t_model.inference_image(t_adv_images_norm)
                t_output_txt = t_model.inference_text(adv_texts_input)
                t_image_feats[images_ids] = t_output_img['image_feat'].cpu().detach()
                t_image_embeds[images_ids] = t_output_img['image_embed'].cpu().detach()
                t_text_feats[texts_ids] = t_output_txt['text_feat'].cpu().detach()
                t_text_embeds[texts_ids] = t_output_txt['text_embed'].cpu().detach()
                t_text_atts[texts_ids] = adv_texts_input.attention_mask.cpu().detach()

                
    s_score_matrix_i2t, s_score_matrix_t2i = retrieval_score(model, s_image_feats, s_image_embeds, s_text_feats,
                                                         s_text_embeds, s_text_atts, num_image, num_text, device=device)

    t_score_matrix_i2t, t_score_matrix_t2i = retrieval_score(t_model, t_image_feats, t_image_embeds, t_text_feats,
                                                         t_text_embeds, t_text_atts, num_image, num_text, device=device)

    if export_adv:
        write_vlr_manifest(save_dir, adv_records)

    return s_score_matrix_i2t.cpu().numpy(), s_score_matrix_t2i.cpu().numpy(), \
        t_score_matrix_i2t.cpu().numpy(), t_score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def retrieval_score(model, image_feats, image_embeds, text_feats, text_embeds, text_atts, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'

    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device),
                                    attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t, score_matrix_t2i



@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img, model_name, rank_index_path):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)


    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]
    
    original_rank_index_path = rank_index_path
    origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_tr1_rank_index.npy')
    origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_tr5_rank_index.npy')
    origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_tr10_rank_index.npy')

    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2) # 在原来的分类成功的样本里，但是现在不在攻击后的成功分类集合里
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)



    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]


    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]

    origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_ir1_rank_index.npy')
    origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_ir5_rank_index.npy')
    origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_ir10_rank_index.npy')

    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2) 
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2)
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)


    eval_result = {'txt_r1_ASR (txt_r1)': f'{asr_tr1}({tr1})',
                   'txt_r5_ASR (txt_r5)': f'{asr_tr5}({tr5})',
                   'txt_r10_ASR (txt_r10)': f'{asr_tr10}({tr10})',
                   'img_r1_ASR (img_r1)': f'{asr_ir1}({ir1})',
                   'img_r5_ASR (img_r5)': f'{asr_ir5}({ir5})',
                   'img_r10_ASR (img_r10)': f'{asr_ir10}({ir10})'}
    return eval_result


def _sga_adv_accuracy(value):
    import re

    match = re.search(r"\(([\d.]+)\)", str(value))
    if match:
        return float(match.group(1))
    return float(value)


def _write_sga_result(path, result):
    txt_r1 = _sga_adv_accuracy(result["txt_r1_ASR (txt_r1)"])
    img_r1 = _sga_adv_accuracy(result["img_r1_ASR (img_r1)"])
    txt_r5 = _sga_adv_accuracy(result["txt_r5_ASR (txt_r5)"])
    img_r5 = _sga_adv_accuracy(result["img_r5_ASR (img_r5)"])
    txt_r10 = _sga_adv_accuracy(result["txt_r10_ASR (txt_r10)"])
    img_r10 = _sga_adv_accuracy(result["img_r10_ASR (img_r10)"])
    payload = {
        "txt_r1": txt_r1,
        "txt_r5": txt_r5,
        "txt_r10": txt_r10,
        "img_r1": img_r1,
        "img_r5": img_r5,
        "img_r10": img_r10,
        "txt_r_mean": (txt_r1 + txt_r5 + txt_r10) / 3.0,
        "img_r_mean": (img_r1 + img_r5 + img_r10) / 3.0,
        "r_mean": (txt_r1 + txt_r5 + txt_r10 + img_r1 + img_r5 + img_r10) / 6.0,
        "avg_sim": 0.0,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model(model_name, model_ckpt, text_encoder, device):
    if model_name not in ("ALBEF", "TCL"):
        raise ValueError(f"SGA 白盒仅支持 ALBEF/TCL，收到 {model_name}")
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)
    model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
    checkpoint = torch.load(model_ckpt, map_location="cpu", weights_only=False)

    try:
        state_dict = checkpoint["model"]
    except Exception:
        state_dict = checkpoint

    if model_name == "TCL":
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )
        state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped

    for key in list(state_dict.keys()):
        if "bert" in key:
            encoder_key = key.replace("bert.", "")
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model, ref_model, tokenizer

def eval_asr(model, ref_model, tokenizer, t_model, t_ref_model, t_tokenizer, t_test_transform, data_loader, device, args, config):
    same_model = (
        args.source_model == args.target_model
        and os.path.abspath(args.source_ckpt) == os.path.abspath(args.target_ckpt)
    )
    if same_model:
        t_model = model
        t_ref_model = ref_model
        t_tokenizer = tokenizer

    model = model.to(device)
    ref_model = ref_model.to(device)

    if not same_model:
        t_model = t_model.to(device)
        t_ref_model = t_ref_model.to(device)

    print("Start eval")
    start_time = time.time()

    scales = None
    if args.scales:
        scales = [float(itm) for itm in args.scales.split(",")]

    save_dir = getattr(args, "save_dir", "") or getattr(args, "output_dir", "")

    score_i2t, score_t2i, t_score_i2t, t_score_t2i = retrieval_eval(
        model,
        ref_model,
        t_model,
        t_ref_model,
        t_test_transform,
        data_loader,
        tokenizer,
        t_tokenizer,
        device,
        config,
        scales=scales,
        save_dir=save_dir,
    )
    

    rank_path = args.original_rank_index_path
    t_result = itm_eval(
        t_score_i2t,
        t_score_t2i,
        data_loader.dataset.img2txt,
        data_loader.dataset.txt2img,
        "TCL",
        rank_path,
    )
    print("Performance on {}: \n {}".format(args.target_model, t_result))

    result = itm_eval(
        score_i2t,
        score_t2i,
        data_loader.dataset.img2txt,
        data_loader.dataset.txt2img,
        "ALBEF",
        rank_path,
    )

    print("Performance on {}: \n {}".format(args.source_model, result))
    print("Performance on {}: \n {}".format(args.target_model, t_result))

    if getattr(args, "result_json", ""):
        _write_sga_result(args.result_json, result)
    

    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))    



def main(args, config):
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    
    print("Creating Source Model")
    model, ref_model, tokenizer = load_model(
        args.source_model, args.source_ckpt, args.source_text_encoder, device
    )
    same_model = (
        args.source_model == args.target_model
        and os.path.abspath(args.source_ckpt) == os.path.abspath(args.target_ckpt)
    )
    if same_model:
        print("Source/target 相同，复用同一模型以节省显存")
        t_model, t_ref_model, t_tokenizer = model, ref_model, tokenizer
    else:
        t_model, t_ref_model, t_tokenizer = load_model(
            args.target_model, args.target_ckpt, args.target_text_encoder, device
        )
 
    #### Dataset ####
    print("Creating dataset")
   
    s_test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),        
    ])

    t_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),  
    ])
    
    test_dataset = paired_dataset(config['test_file'], s_test_transform, config['image_root'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=0, collate_fn=test_dataset.collate_fn)

    eval_asr(model, ref_model, tokenizer, t_model, t_ref_model, t_tokenizer, t_test_transform, test_loader, device, args, config)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--source_model', default='ALBEF', type=str)
    parser.add_argument('--source_text_encoder', default='bert-base-uncased', type=str)
    parser.add_argument('--source_ckpt', default='./checkpoint/albef/flickr30k.pth', type=str)    
    
    parser.add_argument('--target_model', default='TCL', type=str)
    parser.add_argument('--target_text_encoder', default='bert-base-uncased', type=str)
    parser.add_argument('--target_ckpt', default='./checkpoint/tcl/flickr30k.pth', type=str)    
 
    parser.add_argument("--original_rank_index_path", default="./std_eval_idx/flickr30k/")
    parser.add_argument("--scales", type=str, default="0.5,0.75,1.25,1.5")
    parser.add_argument("--result_json", default="")
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    if not args.save_dir:
        args.save_dir = args.output_dir

    yaml_parser = yaml.YAML(typ="safe")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml_parser.load(f)

    main(args, config)


