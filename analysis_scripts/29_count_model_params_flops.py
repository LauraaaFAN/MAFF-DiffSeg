from pathlib import Path
import sys
import torch
import torch.nn as nn

BTCV_ROOT = Path("/data/oldproject/exp/Diff-UNet-main2/BTCV")
sys.path.insert(0, str(BTCV_ROOT))

from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe


class DiffUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_model = BasicUNetEncoder(
            3, 1, 2, [64, 64, 128, 256, 512, 64]
        )
        self.model = BasicUNetDe(
            3, 14, 13, [64, 64, 128, 256, 512, 64]
        )

    def forward(self, image, x, t):
        embeddings = self.embed_model(image)
        out = self.model(x, t, image=image, embeddings=embeddings)
        return out


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_params(name, model):
    total, trainable = count_params(model)
    print(f"{name}")
    print("-" * len(name))
    print(f"Total parameters:     {total:,} ({total / 1e6:.4f} M)")
    print(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.4f} M)")
    print()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffUNet().to(device).eval()

    print_params("BasicUNetEncoder", model.embed_model)
    print_params("BasicUNetDe", model.model)
    print_params("Complete DiffUNet", model)

    # sanity forward
    with torch.no_grad():
        image = torch.randn(1, 1, 96, 96, 96, device=device)
        x = torch.randn(1, 13, 96, 96, 96, device=device)
        t = torch.tensor([10], device=device).long()
        y = model(image, x, t)
        print("Sanity forward output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()