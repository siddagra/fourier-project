

import argparse

import torch

from PIL import Image
from time import time
from FiT import FiT
import json
from types import SimpleNamespace
from timm.models.vision_transformer import _create_vision_transformer


@torch.inference_mode()
def main():
    with open("config.json", "r") as f:
        config = SimpleNamespace(** json.load(f))

    model_kwargs = dict(patch_size=4, embed_dim=768,
                        depth=12, num_heads=12, img_size=[224, 224])
    model = _create_vision_transformer(
        "vit-cifar10", False, **model_kwargs).to("cuda").eval()

    dummy_image = torch.randn(1, 3, 224, 224, device='cuda')
    dummy_image = torch.nn.functional.normalize(dummy_image, 0.5)

    for param in model.parameters():
        param.grad = None
    total = 0.0
    num = 1000
    with torch.no_grad():
        for i in range(30):  # warmup
            _ = model(dummy_image)
        for fname in range(0, num):
            t1 = time()
            _ = model(dummy_image)
            total += time() - t1
    print('num:{} total_time:{}s avg_time:{}s'.format(num, total, total / num))


if __name__ == '__main__':
    main()
