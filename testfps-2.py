

import argparse

import torch

from PIL import Image
from time import time
from FiT import FiT
import json
from types import SimpleNamespace


@torch.inference_mode()
def main():
    with open("config.json", "r") as f:
        config = SimpleNamespace(** json.load(f))

    model = FiT(config).to('cuda').eval()
    print(len([e for e in model.parameters()]))

    dummy_image = torch.randn(1, 3, 1080, 1080, device='cuda')
    dummy_image = torch.nn.functional.normalize(dummy_image, 0.5)

    for param in model.parameters():
        param.grad = None
    total = 0.0
    num = 200
    with torch.no_grad():
        for i in range(100):  # warmup
            _ = model(dummy_image)
        for fname in range(0, num):
            t1 = time()
            _ = model(dummy_image)
            total += time() - t1
        print('num:{} total_time:{}s avg_time:{}s'.format(
            num, total, total / num))


if __name__ == '__main__':
    main()
