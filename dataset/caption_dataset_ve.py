import json
import os

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from dataset.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ve_dataset_attack(Dataset):
    def __init__(self, ann_file, transform, image_root, args, max_words=30):
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        item = self.ann[index]

        img_name = str(item["image"])
        if not img_name.endswith(".jpg"):
            img_name += ".jpg"

        image_path = os.path.join(self.image_root, img_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text = pre_caption(item["sentence"], self.max_words)
        label = self.label_map[item["label"]]

        return image, text, label
