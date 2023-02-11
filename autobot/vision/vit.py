# %%
import torch

from typing import Tuple
from torch import nn

from .base import ImageEmbedding
from ..base import EncoderBlock


class VisionTransformer(nn.Module):

    def __init__(self,
        patch_size,
        emb_dim: int, attn_dim: int, num_heads: int, 
        num_layers: int,
        num_classes: int,
        img_size: Tuple[int, int],  # H, W
        drop_prob=0.1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.emb = ImageEmbedding(emb_dim, patch_size)
        n_patch = int(img_size[0]//patch_size)
        self.position_embeddings = nn.Parameter(torch.randn(1, (n_patch*n_patch) + 1, emb_dim))
        self.emb_dropout = nn.Dropout(drop_prob)

        self.transformer = nn.ModuleList([
            EncoderBlock(emb_dim, attn_dim, num_heads, drop_prob)
            for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(emb_dim, eps=1e-12)
        self.classifier =  nn.Linear(emb_dim, num_classes)

    def forward(self, img):
        B, _, H, W = img.shape  # B, C, H, W

        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, self.emb(img)), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.emb_dropout(embeddings)
        
        x = embeddings
        for layer in self.transformer:
            x = layer(x)
        x = self.layernorm(x)
        logits = self.classifier(x[:, 0, :])
        return logits

    @classmethod
    def from_hugging_face(
        cls,
        patch_size: int,
        emb_dim: int, num_heads: int,
        num_layers: int,
        num_classes: int,
        img_size: Tuple[int, int]
    ):
        from transformers import ViTForImageClassification

        model = ViTForImageClassification.from_pretrained(f'google/vit-base-patch{patch_size}-{img_size[0]}')
        custom = cls(
            patch_size=patch_size,
            emb_dim=emb_dim, attn_dim=emb_dim, num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            img_size=img_size
        )

        children = list(model.children())
        linear = children[1]
        custom.classifier.load_state_dict(linear.state_dict())
        
        vit = list(children[0].children())
        
        vit_emb = vit[0]
        custom.cls_token = vit_emb.cls_token
        custom.position_embeddings = vit_emb.position_embeddings
        custom.emb.projection.load_state_dict(vit_emb.patch_embeddings.projection.state_dict())
        
        for i, (origin, layer) in enumerate(zip(list(vit[1].children())[0], custom.transformer)):
            layer.ff.linear_1.load_state_dict(origin.intermediate.dense.state_dict())
            layer.ff.linear_2.load_state_dict(origin.output.dense.state_dict())

            layer.mha_norm.load_state_dict(origin.layernorm_before.state_dict())
            layer.ff_norm.load_state_dict(origin.layernorm_after.state_dict())

            layer.mha.query.load_state_dict(origin.attention.attention.query.state_dict())
            layer.mha.key.load_state_dict(origin.attention.attention.key.state_dict())
            layer.mha.value.load_state_dict(origin.attention.attention.value.state_dict())
            layer.mha.linear.load_state_dict(origin.attention.output.dense.state_dict())
            
        custom.layernorm.load_state_dict(vit[2].state_dict())

        return custom
