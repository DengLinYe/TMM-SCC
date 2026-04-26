import json
import random
from collections import Counter

original_files = [
    "./data/flickr30k/flickr30k_test.json",
    "./data/snli-ve/ve_train.json",
    "./data/snli-ve/ve_test.json",
    "./data/snli-ve/ve_dev.json",
]
mini_files = [
    "./data/flickr30k/flickr30k_test_mini.json",
    "./data/snli-ve/ve_train_mini.json",
    "./data/snli-ve/ve_test_mini.json",
    "./data/snli-ve/ve_dev_mini.json",
]


def sample_by_image(file_path, save_path, target_img_count=100):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_to_data = {}
    for entry in data:
        img_id = entry.get("image") or entry.get("image_id")
        if img_id not in img_to_data:
            img_to_data[img_id] = []
        img_to_data[img_id].append(entry)

    all_images = list(img_to_data.keys())
    sampled_images = random.sample(all_images, min(target_img_count, len(all_images)))

    mini_data = []
    for img_id in sampled_images:
        mini_data.extend(img_to_data[img_id])

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mini_data, f, indent=4)

    print(f"File: {file_path}")
    print(f"  - Unique Images: {len(sampled_images)}")
    print(f"  - Total Entries: {len(mini_data)}")

    if "ve_" in file_path:
        labels = [entry["label"] for entry in mini_data if "label" in entry]
        print(f"  - Label Distribution: {Counter(labels)}")
    print("-" * 30)


for i in range(len(original_files)):
    sample_by_image(original_files[i], mini_files[i])
