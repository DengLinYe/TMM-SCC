from enum import Enum

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
from torchvision import transforms

images_normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)


class NormType(Enum):
    Linf = 0
    L2 = 1


class ImageProcessing:
    def __init__(self):
        self.images_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

    def gkern(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def clamp_by_l2(self, x, max_norm):
        norm = torch.norm(x, dim=(1, 2, 3), p=2, keepdim=True)
        factor = torch.min(max_norm / norm, torch.ones_like(norm))
        return x * factor

    def clamp_by_l1(self, x, max_norm):
        norm = torch.norm(x, dim=(1, 2, 3), p=1, keepdim=True)
        factor = torch.min(max_norm / norm, torch.ones_like(norm))
        return x * factor

    def random_init(self, x, norm_type, epsilon):
        delta = torch.zeros_like(x)
        if norm_type == NormType.Linf:
            delta.data.uniform_(0.0, 1.0)
            delta.data = delta.data * epsilon
        elif norm_type == NormType.L2:
            delta.data.uniform_(0.0, 1.0)
            delta.data = delta.data - x
            delta.data = self.clamp_by_l2(delta.data, epsilon)
        elif norm_type == NormType.L1:
            delta.data.uniform_(0.0, 1.0)
            delta.data = delta.data - x
            delta.data = self.clamp_by_l1(delta.data, epsilon)
        return delta

    def get_kernel(self, device, kernel_size=15):
        sigma = kernel_size / 2.0
        grid = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        gaussian = torch.exp(-(grid**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        kernel2d = gaussian.unsqueeze(1) * gaussian.unsqueeze(0)
        kernel = kernel2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).to(device)
        return kernel


class ImageAttacker:
    def __init__(
        self,
        epsilon,
        args,
        norm_type=NormType.Linf,
        random_init=True,
        cls=True,
        **kwargs,
    ):
        self.norm_type = norm_type
        self.random_init = random_init
        self.epsilon = epsilon
        self.cls = cls
        self.args = args
        self.image_processing = ImageProcessing()
        self.preprocess = None

    def input_diversity(self, image, prob=0.5):
        if torch.rand(1) > prob:
            return image

        img_size = image.shape[-1]
        scale_res = getattr(self.args, "scale_res", 1.1)
        img_resize = int(img_size * scale_res)

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,)).item()
        rescaled = F.interpolate(
            image, size=[rnd, rnd], mode="bilinear", align_corners=False
        )

        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem, size=(1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem, size=(1,)).item()
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0)

        return F.interpolate(
            padded, size=[img_size, img_size], mode="bilinear", align_corners=False
        )

    def attack(self, image, attMap, num_iters):
        device = image.device

        if attMap.shape[-1] != image.shape[-1]:
            attMap = torch.zeros_like(image).to(device)

        epsilon_per_val = getattr(self.args, "epsilon_per", 0.5)
        att_mask_val = getattr(self.args, "att_mask", 0.0)

        epsilon = self.epsilon * epsilon_per_val
        eosilon_att = self.epsilon - epsilon
        att_sum = (attMap > att_mask_val).sum().float()
        total_pixels = float(image.shape[1] * image.shape[2] * image.shape[3])

        if att_sum == 0:
            epsilon_att = 0.0
        else:
            epsilon_att = (eosilon_att / (att_sum / total_pixels)).item()

        eps = epsilon / 255.0
        eps_att = epsilon_att / 255.0
        if self.random_init:
            self.delta = self.image_processing.random_init(
                image, self.norm_type, eps
            ).to(device)
            self.delta_att = self.image_processing.random_init(
                attMap, self.norm_type, eps_att
            ).to(device)
            self.g = torch.zeros_like(image).to(device)
            self.g_att = torch.zeros_like(attMap).to(device)
        else:
            self.delta = torch.zeros_like(image)

        steps = getattr(self.args, "num_steps", getattr(self.args, "steps", 10))
        if steps == 0:
            steps = 10
        epsilon_per = eps / steps * 1.25
        epsilon_per_att = eps_att / steps * 1.25

        img_size = image.shape[-1]

        for i in range(num_iters):
            self.delta = self.delta.detach()
            self.delta_att = self.delta_att.detach()
            self.delta.requires_grad = True
            self.delta_att.requires_grad = True

            image_diversity = self.input_diversity(
                image
                + self.delta
                + (attMap > self.args.att_mask).to(device) * self.delta_att
            )

            if image_diversity.shape[-1] != img_size:
                image_diversity = F.interpolate(
                    image_diversity, size=(img_size, img_size), mode="bilinear"
                )

            if self.preprocess is not None:
                image_diversity = self.preprocess(image_diversity)

            yield image_diversity

            if self.delta.grad is None:
                print(
                    "Warning: self.delta.grad is None. Check if backward() is called."
                )
                continue

            grad = self.delta.grad.clone()
            grad_att = self.delta_att.grad.clone()

            k_size = getattr(self.args, "kernel_size", 15)

            grad = F.conv2d(
                grad,
                weight=self.image_processing.get_kernel(device, k_size),
                stride=(1, 1),
                groups=3,
                padding=(k_size - 1) // 2,
            )
            grad_att = F.conv2d(
                grad_att,
                weight=self.image_processing.get_kernel(device, k_size),
                stride=(1, 1),
                groups=3,
                padding=(k_size - 1) // 2,
            )
            noise = grad / torch.abs(grad).mean(dim=(1, 2, 3), keepdim=True)
            noise_att = grad_att / torch.abs(grad_att).mean(dim=(1, 2, 3), keepdim=True)
            self.g = self.g * self.args.momentum + noise
            self.g_att = self.g_att * self.args.momentum + noise_att

            self.delta = self.delta.data + epsilon_per * torch.sign(self.g)
            self.delta_att = self.delta_att.data + epsilon_per_att * torch.sign(
                self.g_att
            )

            self.delta = torch.clamp(self.delta, -eps, eps)
            self.delta_att = torch.clamp(self.delta_att, -eps_att, eps_att)

        yield (
            image
            + self.delta
            + (attMap > self.args.att_mask).to(device) * self.delta_att
        ).detach()

    def get_grad(self):
        self.grad = self.delta.grad.clone()
        return self.grad

    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return self.image_processing.clamp_by_l2(delta, epsilon)

    def normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)
