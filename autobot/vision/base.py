import typing
from torch import nn


class ImageEmbedding(nn.Module):
    '''
        Convert an image into a sequences of img_token
    '''

    def __init__(self, emb_dim: int, patch_size):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, typing.Iterable) else (patch_size, patch_size)

        self.projection = nn.Conv2d(
            3, emb_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        self.name = f'{patch_size[0]}_{patch_size[1]}'

    def forward(self, img):
        '''
            Args:
                img (B, 3, H, W)

            Returns:
                img_emb (B, H*W, D)
        '''
        print(self.projection(img).shape)
        print(self.projection(img).flatten(2).shape)
        return self.projection(img).flatten(2).transpose(1, 2)
