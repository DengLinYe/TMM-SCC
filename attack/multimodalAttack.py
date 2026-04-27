import numpy as np
import torch
from pytorch_msssim import ssim
from torchvision import transforms

from .imageAttack import ImageAttacker

images_normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)


def equal_normalize(x):
    return x


gx = 1
cx = []


class MultiModalAttacker:
    def __init__(self, net, text_attacker, tokenizer, args, cls=True):
        self.net = net
        self.text_attacker = text_attacker
        self.tokenizer = tokenizer
        self.args = args
        self.cls = cls
        if hasattr(text_attacker, "sample_batch_size"):
            self.sample_batch_size = text_attacker.sample_batch_size

        self.repeat = 1
        self.image_normalize = images_normalize
        self.image_attacker = ImageAttacker(epsilon=args.epsilon, args=args, cls=cls)
        self.ssim = ssim

    def get_origin_and_adv_embeds(self, images, text, device, max_length, k):
        with torch.no_grad():
            text_input = self.tokenizer(
                text * self.repeat,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            origin_output = self.net.inference(
                self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
                text_input,
                use_embeds=False,
            )
            if self.cls:
                origin_embeds = origin_output["fusion_output"][:, 0, :].detach()
            else:
                origin_embeds = origin_output["fusion_output"].flatten(1).detach()

        with torch.no_grad():
            text_adv = self.text_attacker.attack(self.net, images, text, k)
            if not text_adv:
                print("警告：大模型请求失败，使用原句作为对抗文本")
                text_adv = text

            text_adv_input = self.tokenizer(
                text_adv,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            text_adv_output = self.net.inference(
                self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
                text_adv_input,
                use_embeds=False,
            )
            if self.cls:
                text_adv_embed = text_adv_output["fusion_output"][:, 0, :].detach()
            else:
                text_adv_embed = text_adv_output["fusion_output"].flatten(1).detach()

        return (
            origin_embeds,
            text_adv_embed,
            text_input,
            origin_output,
            text_adv_input,
            text_adv,
        )

    def getGradCam(self, gradcam, shape):
        """
        使用 PyTorch 原生插值替代 OpenCV，支持任意通道数且不占用系统内存。
        gradcam: [H, W, C] 类型的张量
        shape: (target_H, target_W)
        """
        import torch.nn.functional as F

        if not torch.is_tensor(gradcam):
            gradcam = torch.from_numpy(gradcam).to(self.net.device)

        x = gradcam.permute(2, 0, 1).unsqueeze(0)

        x = F.interpolate(x, size=shape, mode="bilinear", align_corners=False)

        x_min = x.min()
        x_max = x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = x - x_min

        return x.squeeze(0).permute(1, 2, 0)

    def getAtt(self, attImage, text_input, device, indices, args):
        if attImage.size(0) == 0:
            return torch.zeros((1, 3, 224, 224)).to(device)
        self.net.train()

        with torch.enable_grad():
            attImage_with_grad = attImage.clone().detach().requires_grad_(True)

            image_embeds = self.net.visual_encoder(images_normalize(attImage_with_grad))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            text_batch_size = text_input.input_ids.size(0)

            if image_embeds.size(0) != text_batch_size:
                image_embeds = image_embeds.repeat(text_batch_size, 1, 1)
                image_atts = image_atts.repeat(text_batch_size, 1)

            output = self.net.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            vl_embeddings = output.last_hidden_state[:, 0, :]

            if hasattr(self.net, "itm_head"):
                vl_output = self.net.itm_head(vl_embeddings)
                loss_pre = vl_output[:, 1].sum()
            else:
                vl_output = self.net.cls_head(vl_embeddings)
                loss_pre = vl_output.max(dim=1)[0].sum()

            self.net.zero_grad()
            loss_pre.backward()

        with torch.no_grad():
            cam = []

            text_batch_size = text_input.attention_mask.size(0)

            mask_att = text_input.attention_mask.view(
                text_input.attention_mask.size(0), 1, -1, 1, 1
            )

            for blk in range(
                6, len(self.net.text_encoder.base_model.base_model.encoder.layer)
            ):
                grad_data = self.net.text_encoder.base_model.base_model.encoder.layer[
                    blk
                ].crossattention.self.get_attn_gradients()

                cams_data = self.net.text_encoder.base_model.base_model.encoder.layer[
                    blk
                ].crossattention.self.get_attention_map()

                if grad_data is None:
                    continue

                grads = grad_data.detach()[:, :, :, 1:].reshape(
                    text_batch_size, 12, -1, 24, 24
                )

                cams = (
                    cams_data.detach()[:, :, :, 1:]
                    .clamp(min=0)
                    .reshape(text_batch_size, 12, -1, 24, 24)
                )

                gradcam_layer = cams * grads
                gradcam_layer = gradcam_layer.mean(1).mean(1)
                cam.append(gradcam_layer)

            if len(cam) == 0:
                return torch.zeros((1, 3, 224, 224)).to(device)

            gradcam_final = 0
            for i in range(len(cam)):
                gradcam_final += cam[i]

            gradcam_final = gradcam_final / attImage.size(0)
            gradcam_final = gradcam_final.permute(1, 2, 0)

            gradcam_final = self.getGradCam(gradcam_final, attImage[0].shape[1:3])
            gradcam_final = gradcam_final.detach().cpu().numpy()
            gradcam_final = gradcam_final.transpose(2, 0, 1)

        if args.dataset == "snli-ve":
            attMap = (
                torch.from_numpy(gradcam_final)
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
                .to(device)
            )
        else:
            attMap = np.add.reduceat(gradcam_final, [0, indices[0]])[::2]
            attMap = (attMap / 5).squeeze()
            attMap = torch.repeat_interleave(
                torch.tensor(attMap).unsqueeze(0).unsqueeze(0), repeats=3, dim=1
            )

        return attMap

    def run_transfer_attack(self, images, text, args, num_iters, k=10, max_length=30):
        device = images.device

        (
            origin_embeds,
            _,
            text_input,
            origin_output,
            _,
            _,
        ) = self.get_origin_and_adv_embeds(images, text, device, max_length, k)

        loss_fn = torch.nn.CosineEmbeddingLoss()

        if args.dataset == "snli-ve":
            indices = torch.arange(images.shape[0]).cpu()
            attImage = images.to(device)
        else:
            indices = torch.linspace(
                4, images.shape[0] - 1, int(images.shape[0] / 5), dtype=int
            ).cpu()
            attImage = images.index_select(0, indices.to(device))

        attMap = self.getAtt(attImage, text_input, device, indices, args)
        image_attack = self.image_attacker.attack(attImage, attMap, num_iters)

        if hasattr(self.args, "text_method") and self.args.text_method == "scc":
            _, _, _, _, scc_text_adv_input, scc_text_adv = (
                self.get_origin_and_adv_embeds(images, text, device, max_length, k)
            )

        for i in range(num_iters):
            image_diversity = next(image_attack)

            if args.dataset == "snli-ve":
                adv = image_diversity
            else:
                adv = [
                    image_diversity[j].repeat(5, 1, 1, 1)
                    for j in range(image_diversity.shape[0])
                ]
                if len(adv) > 0:
                    adv = torch.cat(adv, dim=0)
                else:
                    adv = images

            if hasattr(self.args, "text_method") and self.args.text_method == "scc":
                text_adv_input = scc_text_adv_input
                text_adv = scc_text_adv
            else:
                _, _, _, _, text_adv_input, text_adv = self.get_origin_and_adv_embeds(
                    adv.detach(), text, device, max_length, k
                )

            adv_output = self.net.inference(adv, text_adv_input, use_embeds=False)

            with torch.no_grad():
                vdt = self.net.inference(adv.detach(), text_input, use_embeds=False)
                text_adv_output = self.net.inference(
                    self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
                    text_adv_input,
                    use_embeds=False,
                )

            if args.cls:
                adv_embed = adv_output["fusion_output"][:, 0, :]
                vdt_embed = vdt["fusion_output"][:, 0, :]
                text_adv_embed = text_adv_output["fusion_output"][:, 0, :].detach()
            else:
                adv_embed = adv_output["fusion_output"].flatten(1)
                vdt_embed = vdt["fusion_output"].flatten(1)
                text_adv_embed = text_adv_output["fusion_output"].flatten(1).detach()

            y = torch.ones(adv_embed.shape[0]).to(device)
            cos_ao_loss_nosoft = loss_fn(adv_embed, origin_embeds, y)
            cos_ot_nosoft = loss_fn(origin_embeds, text_adv_embed, y)
            cos_ov_nosoft = loss_fn(origin_embeds, vdt_embed, y)
            adv_origin_img_ssim_loss = 1 - self.ssim(
                adv[0].unsqueeze(0), images[0].unsqueeze(0), data_range=1.0
            )
            loss = (
                cos_ao_loss_nosoft
                + cos_ot_nosoft
                + cos_ov_nosoft
                + 10 * adv_origin_img_ssim_loss
            )
            self.net.zero_grad()
            loss.backward()

            torch.cuda.empty_cache()

        images_adv = next(image_attack)

        if args.dataset != "snli-ve":
            images_adv = [
                images_adv[j].repeat(5, 1, 1, 1) for j in range(images_adv.shape[0])
            ]

            images_adv = torch.cat(images_adv, dim=0)

        if not (hasattr(self.args, "text_method") and self.args.text_method == "scc"):
            _, _, _, _, _, text_adv = self.get_origin_and_adv_embeds(
                adv, text, device, max_length, k
            )

        return images_adv, text_adv
