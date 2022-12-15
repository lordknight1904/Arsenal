import typing

import torch
from torch import nn


class ImageEmbedding(nn.Module):
    '''
        Convert an image into a sequences of img_token
    '''

    def __init__(self, emb_dim: int, patch_size):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, typing.Iterable) else (patch_size, patch_size)

        self.emb_dim = emb_dim
        self.projection = nn.Conv2d(
            3, emb_dim,
            kernel_size=patch_size, stride=patch_size,
        )  # (B, 3, H, W) --> (B, D, H, W) --> (B, D, H*W) --> (B, H*W, D)

    def forward(self, img):
        '''
            Args:
                img (B, 3, H, W)

            Returns:
                img_emb (B, H*W, D)
        '''
        # return self.projection(img).flatten(2).transpose(1, 2) / torch.sqrt(self.emb_dim
        return self.projection(img).flatten(2).transpose(1, 2)
