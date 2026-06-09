import json
import os

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from dataset.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class pair_dataset_attack(Dataset):
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

            if "image_path" in ann:
                img_name = str(ann["image_path"])
            else:
                img_name = str(ann.get("image") or ann.get("image_id"))
                if not img_name.endswith((".jpg", ".png")):
                    img_name += ".jpg"

            if "adv_text" in ann:
                captions = [ann["adv_text"]]
            elif "caption" in ann:
                captions = ann["caption"]
                if isinstance(captions, str):
                    captions = [captions]
            else:
                captions = [ann.get("sentence", "")]

            for j, caption in enumerate(captions):
                self.image.append(img_name)
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i

                if "image_id" in ann:
                    self.image_ids[txt_id] = str(ann["image_id"])
                else:
                    basename = os.path.basename(img_name)
                    name_without_ext = os.path.splitext(basename)[0]

                    if args.dataset == "flickr":
                        self.image_ids[txt_id] = name_without_ext
                    elif args.dataset == "mscoco":
                        if "_" in name_without_ext:
                            self.image_ids[txt_id] = name_without_ext.split("_")[-1]
                        else:
                            self.image_ids[txt_id] = name_without_ext

                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        text = self.text[index]
        image_id = self.image_ids[index]

        return image, text, index, image_id
