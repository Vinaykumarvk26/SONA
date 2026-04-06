import torch
import torch.nn as nn
import torchvision.models as models


class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y)
        x = x + attn_out
        z = self.norm2(x)
        x = x + self.mlp(z)
        return x


class FERHybridNet(nn.Module):
    def __init__(self, num_classes: int = 7, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet34":
            base = models.resnet34(weights=None)
            feat_dim = 512
        else:
            base = models.resnet18(weights=None)
            feat_dim = 512

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.patch_proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        self.vit_block = VisionTransformerBlock(embed_dim=feat_dim, num_heads=8)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.patch_proj(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.vit_block(x)
        x = x.mean(dim=1)
        return self.head(x)
