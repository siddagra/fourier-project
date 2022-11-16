import re
from scipy import linalg
import torch.utils.checkpoint
import torch
from torch import nn
from torch import linalg
from pytorch_lightning import LightningModule


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        img_size = config.img_size
        patch_size = config.patch_size
        grid_size = (img_size[0] // patch_size[0],
                     img_size[1] // patch_size[1])
        num_patches = grid_size[0] * grid_size[1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))

        self.proj = nn.Conv2d(
            config.in_chans, config.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, num_patches+1, config.embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.positional_embeddings
        x = self.norm(x)
        return x  # B, C, P + 1


class FFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        return torch.fft.fft(torch.fft.fft(x.float(), dim=-1), dim=-2).real


class FiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fft = FFTLayer()
        self.layerNorm1 = nn.LayerNorm(
            config.embed_dim, eps=1e-12)
        self.ff = nn.Linear(
            config.embed_dim, config.dim_feedforward)
        self.dense = nn.Linear(
            config.dim_feedforward, config.embed_dim)
        self.layerNorm2 = nn.LayerNorm(
            config.embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        fftOut = self.fft(x)
        x = self.layerNorm1(fftOut + x)
        x = self.ff(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.layerNorm2(x + fftOut)
        return x


class FiTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([FiTBlock(config)
                                   for _ in range(config.num_layers)])

    def forward(self, x):
        for i, layer_module in enumerate(self.layer):
            x = layer_module(x)

        return x


class FiTHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim, config.num_classes)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.dense(x[:, 0])
        out = self.activation(out)
        return out


class FiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed = PatchEmbeddings(config)
        self.encoder = FiTEncoder(config)
        self.head = FiTHead(config)

    def forward_features(self, x):
        embeds = self.embed(x)
        features = self.encoder(embeds)
        return features

    def forward(self, x):
        embeds = self.embed(x)
        features = self.encoder(embeds)
        out = self.head(features)
        return features, out
