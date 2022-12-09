import typing
from torch import nn


class ClassificationHead(nn.Module):
    
    def __init__(self,
        emb_dim,
        num_classes,
    ):
        super().__init__()


    def forward(self, x):
        return x
    
class VisionTransformer(nn.Module):

    def __init__(self):
        super().__init__()
