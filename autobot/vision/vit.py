# %%
from typing import Tuple
from torch import nn

from ..base import RelativeTransformer
from .base import ImageEmbedding


class ClassificationHead(nn.Module):
    
    def __init__(self,
        emb_dim,
        num_classes,
    ):
        super().__init__()

        self.linear = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    

class VisionTransformer(nn.Module):

    def __init__(self,
        patch_size,
        emb_dim: int, num_heads: int, 
        num_classes: int,
        img_size: Tuple[int, int]  # H, W
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size

        self.emb = ImageEmbedding(emb_dim, patch_size)
        self.transformer = RelativeTransformer(
            emb_dim=emb_dim, attn_dim=emb_dim, v_dim=emb_dim,
            num_heads=num_heads
        )
        self.classifier = ClassificationHead(emb_dim, num_classes)

    def forward(self, img):
        B, _, H, W = img.shape  # B, C, H, W

        emb = self.emb(img)
        
        # x = self.transformer(emb)

        return emb
