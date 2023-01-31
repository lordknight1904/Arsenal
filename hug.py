# %%
import h5py
from functools import cached_property
from dataclasses import dataclass
from transformers import ViTForImageClassification
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTImageProcessor

from pathlib import Path as path
from PIL import Image
import csv

import numpy as np
from typing import ClassVar, Tuple

import autobot

# @dataclass(slots=True)
# class InputImage:
#     image: np.ndarray
#     image_dtype: ClassVar[np.dtype]=np.float32

#     @classmethod
#     def _img_from_path(cls, path:str, size, mean=0., std=1.) -> np.ndarray:
#         img = Image.open(path)

#         img = np.array(img, dtype=cls.image_dtype)
#         return img

#     @classmethod
#     def from_path(cls, path: str):
#         return cls(cls._img_from_path(path))

#     def to_disk(self) -> Tuple: 
#         raise NotImplementedError


# class BaseSplit(Dataset):
#     '''
#         A HDF5-based PyTorch Dataset
#             - support for multiple workers (num_worker > 0)
#             - support for asynchronous multiprocessing for pre-process dataset
#     '''

#     def __init__(self, h5py_file, split, name):
#         self.h5py_flie = h5py_file
#         self.split = split
#         self.name = name

#         with h5py_file as f:
#             self.size = f.attrs['size']

#     @cached_property
#     def dataset(self): return h5py.File(self.file_path, 'r')

#     def __len__(self): return self.size

#     def __getitem__(self, idx): 
#         raise NotImplementedError
    

# class ImageNetDataset(Dataset):
    
#     def __init__(self, folder_path, sample_tuple):
#         self.folder_path = folder_path
#         self.sample_tuple = sample_tuple

#         self.mean = np.array([0.5, 0.5, 0.5])
#         self.std = np.array([0.5, 0.5, 0.5])

#     def __len__(self):
#         return len(self.sample_tuple)

#     def __getitem__(self, idx):
#         img_path, label = self.sample_tuple[idx]

#         img = Image.open(path.joinpath(self.folder_path, img_path)).convert('RGB')
#         # encoding = self.feature_extractor(images=img, return_tensors="pt")

#         img = img.resize((384, 384))
#         img = np.array(img) / 255
#         img = (img - self.mean)/self.std
#         # print('img', img.shape, img.min(), img.max())

#         # return encoding['pixel_values'][0], label
#         return img.transpose(2, 0, 1), label
        
#     @classmethod
#     def from_path(cls, folder_path, split):
#         c2i, i2c = {}, {}
        
#         with open(path.joinpath(folder_path, 'LOC_synset_mapping.txt')) as f:
#             for i, line in enumerate(f.readlines()):
#                 segments = line.strip().split(' ')
#                 label_idx = segments[0]
#                 c2i[label_idx] = i
#                 i2c[i] = label_idx

#         def _read_path(row):
#             sample_id = row[0]
#             strings = sample_id.split('_')
#             split = strings[1] if strings[0] == 'ILSVRC2012' else 'train'
        
#             file_path = path('ILSVRC', 'Data', 'CLS-LOC')
#             if split == 'train':
#                 file_path = path.joinpath(file_path, strings[0], strings[1], f'{sample_id}.JPEG')
#             else:
#                 file_path = path.joinpath(file_path, split, f'{sample_id}.JPEG')

#             # label_name, *bboxes = row[1].strip().split(' ')
#             label_name = row[1].strip().split(' ')[0]
#             label_id = c2i[label_name]

#             return (file_path, label_id)

#         with open(path.joinpath(folder_path, 'LOC_val_solution.csv')) as csv_file:
#             dt = csv.reader(csv_file)
#             next(dt)  # skip header row
#             sample_tuple = list(map(_read_path, dt))
#             # for a in dt:
#             #     print(a)
#             #     print(_read_path(a))
#             #     break
#         # print(sample_tuple[0])
#         return cls(folder_path, sample_tuple)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
    model.eval()
    model.to(device)
    count, total = 0, 0
    
    val_ds = autobot.ImageNetSplit.from_path(
        folder_path=path.joinpath(path.home(), 'datasets', 'ImageNet'),
        csv_file='LOC_val_solution.csv'
    )
    # for i, sample in enumerate(val_ds):
    #     image, label = sample['image'], sample['label']
    #     print(image.shape)
    #     print(label)
    #     break
    
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=6)
    # for batch in val_loader:
    #     print(batch['image'].shape)
    #     print(batch['label'].shape)
    #     break

    for i, batch in enumerate(val_loader):
        images, labels = batch['image'], batch['label']
        print(f'\r{i}\{len(val_loader)}', end='')
        with torch.no_grad():
            outputs = model(images.to(device))
        pred = outputs.logits.argmax(-1)
        acc = labels.eq(pred.cpu())
        count += acc.sum().item()
        total += labels.shape[0]

    print(count / total)
