# %%
from typing import Tuple
from torch import nn

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
        num_classes: int
    ):
        super().__init__()

        self.emb = ImageEmbedding(emb_dim, patch_size)
        self.pos = None
        self.transformer = None
        self.classifier = ClassificationHead(emb_dim, num_classes)

    def forward(self, img):
        B, _, H, W = img.shape  # B, C, H, W

        emb = self.emb(img)
        print(emb.shape)

        return emb


if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    vit = VisionTransformer(64, (4, 4), 10)
