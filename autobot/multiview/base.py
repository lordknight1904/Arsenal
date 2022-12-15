import typing

import torch
from torch import nn

from ..base import MultiHeadAttention


class MultiViewEmbedding(nn.Module):

    def __init__(self, emb_dim: int, patch_size):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, typing.Iterable) else (patch_size, patch_size)

        self.emb_dim = emb_dim
        self.projection = nn.Conv2d(
            3, emb_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, img):
        '''
            Args:
                img (B, N, 3, H, W): input images

            Returns:
                img_emb (B, N, H*W, D): a sequence of embedded image patches
        '''
        # return self.projection(img).flatten(2).transpose(1, 2) / torch.sqrt(self.emb_dim
        B, N, D, H, W = img.shape
        img = img.view(B*N, D, H, W)
        emb = self.projection(img)  # (B*N, D, H, W)
        emb = emb.flatten(2)  # (B*N, D, H*W)
        emb = emb.transpose(1, 2)  # (B*N, H*W, D)
        return emb.view(B, N, H*W, self.emb_dim)


class SingleViewAttention(MultiHeadAttention):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        '''
            x: (B, N, D)
        '''
        return x


class MultiViewAttention():
    pass
