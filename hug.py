# %%
import h5py
from functools import cached_property
from dataclasses import dataclass
from transformers import ViTForImageClassification
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTImageProcessor
import transformers.models.vit.modeling_vit
from pathlib import Path as path
from PIL import Image
import csv

import numpy as np
from typing import ClassVar, Tuple

import autobot


if __name__ == '__main__':
    
    val_ds = autobot.ImageNetSplit.from_path(
        folder_path=path.joinpath(path.home(), 'datasets', 'ImageNet'),
        csv_file='LOC_val_solution.csv',
        w=384, h=384,
        use_cache=True
    )
    
    # train_ds = autobot.ImageNetSplit.from_path(
    #     folder_path=path.joinpath(path.home(), 'datasets', 'ImageNet'),
    #     csv_file='LOC_train_solution.csv',
    #     w=224, h=224,
    #     use_cache=False
    # )
    
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=6)
    # for i, batch in enumerate(val_loader):
    #     images, labels = batch['image'], batch['label']
    #     print(images.shape)
    #     print(labels.shape)
    #     break

    device = torch.device('cuda:0')
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
    model = autobot.vit.VisionTransformer.from_hugging_face(
        patch_size=16,
        emb_dim=768, num_heads=12,
        num_layers=12, 
        num_classes=1000,
        img_size=(384, 384)
    )
    model.eval()
    model.to(device)
    count, total = 0, 0

    for i, batch in enumerate(val_loader):
        images, labels = batch['image'], batch['label']
        print(f'\r{i:<3}/{len(val_loader):<3} {100*i/len(val_loader):.0f}%', end='')
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(images.to(device))
            # print('outputs', outputs.shape)
        # pred = outputs.logits.argmax(-1)
        pred = outputs.argmax(-1)
        acc = labels.eq(pred.cpu())
        count += acc.sum().item()
        total += labels.shape[0]

    print()
    print(count / total)
# 
