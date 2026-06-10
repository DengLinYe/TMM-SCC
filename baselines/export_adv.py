import json
import os
from typing import List

import torchvision


def vlr_adv_dir(save_dir: str) -> str:
    return os.path.join(save_dir, "adv_samples_retrieval")


def ve_adv_dir(save_dir: str) -> str:
    return os.path.join(save_dir, "adv_samples_ve")


def append_vlr_records(
    adv_records: List[dict],
    adv_save_dir: str,
    images,
    orig_texts: List[str],
    adv_texts: List[str],
    text_ids,
    step: int,
    batch_size: int,
) -> None:
    os.makedirs(adv_save_dir, exist_ok=True)
    for i in range(len(adv_texts)):
        global_idx = step * batch_size + i
        img_filename = f"adv_{global_idx}.png"
        torchvision.utils.save_image(images[i], os.path.join(adv_save_dir, img_filename))
        tid = text_ids[i].item() if hasattr(text_ids[i], "item") else int(text_ids[i])
        adv_records.append(
            {
                "image_id": global_idx,
                "text_id": tid,
                "image_path": img_filename,
                "orig_text": orig_texts[i],
                "adv_text": adv_texts[i],
            }
        )


def write_vlr_manifest(save_dir: str, adv_records: List[dict]) -> None:
    path = os.path.join(save_dir, "vlr_adv_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(adv_records, f, indent=4, ensure_ascii=False)


def append_ve_records(
    adv_records: List[dict],
    adv_save_dir: str,
    images,
    orig_texts: List[str],
    adv_texts: List[str],
    labels,
    step: int,
    batch_size: int,
) -> None:
    os.makedirs(adv_save_dir, exist_ok=True)
    for i in range(len(adv_texts)):
        global_idx = step * batch_size + i
        img_filename = f"adv_{global_idx}.png"
        torchvision.utils.save_image(images[i], os.path.join(adv_save_dir, img_filename))
        label = labels[i].item() if hasattr(labels[i], "item") else int(labels[i])
        adv_records.append(
            {
                "image_id": global_idx,
                "image_path": img_filename,
                "orig_text": orig_texts[i],
                "adv_text": adv_texts[i],
                "label": label,
            }
        )


def write_ve_manifest(save_dir: str, adv_records: List[dict]) -> None:
    path = os.path.join(save_dir, "ve_adv_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(adv_records, f, indent=4, ensure_ascii=False)


def append_sga_vlr_batch(
    adv_records: List[dict],
    adv_save_dir: str,
    adv_images,
    orig_texts: List[str],
    adv_texts: List[str],
    text_ids,
    txt2img: List[int],
    pair_offset: int,
) -> None:
    os.makedirs(adv_save_dir, exist_ok=True)
    for j in range(len(adv_texts)):
        global_idx = pair_offset + j
        img_idx = txt2img[j]
        img_filename = f"adv_{global_idx}.png"
        torchvision.utils.save_image(
            adv_images[img_idx], os.path.join(adv_save_dir, img_filename)
        )
        tid = text_ids[j].item() if hasattr(text_ids[j], "item") else int(text_ids[j])
        adv_records.append(
            {
                "image_id": global_idx,
                "text_id": tid,
                "image_path": img_filename,
                "orig_text": orig_texts[j],
                "adv_text": adv_texts[j],
            }
        )
