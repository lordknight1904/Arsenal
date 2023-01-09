import typing

import torch
from torch import nn

import csv
import h5py
from dataclasses import dataclass
from typing import ClassVar, Tuple
from pathlib import Path as path
import numpy as np

from ..io import BaseSplit


class ImageNetSplit(BaseSplit):

    @dataclass(slots=True)
    class ClassificationImage(BaseSplit.InputImage):
        label: np.uint
        label_dtype: ClassVar[np.dtype] = np.uint16

        @classmethod
        def from_path(cls, path: str, label: int):
            return cls(
                image=super(ImageNetSplit.ClassificationImage, cls)._img_from_path(path),
                # image=cls._read,
                label=cls.label_dtype(label)  # unit16 gives a maximum of 65535 classes
            )
        
        def to_disk(self) -> Tuple:
            return (self.image.shape[1], self.image.shape[0], self.image.flatten(), self.label,)

    @classmethod
    def from_path(cls, folder_path, split):
        h5py_path = path.joinpath(folder_path, f'{split}.h5')

        custom_dict = cls.IOMapping()

        with open(path.joinpath(folder_path, 'LOC_synset_mapping.txt')) as f:
            for i, line in enumerate(f.readlines()):
                segments = line.strip().split(' ')
                label_idx = segments[0]
                custom_dict[label_idx] = i

        def _process_row(row):
            sample_id = row[0]

            strings = sample_id.split('_')
            split = strings[1] if strings[0] == 'ILSVRC2012' else 'train'
        
            file_path = path.joinpath(folder_path, 'ILSVRC', 'Data', 'CLS-LOC')
            if split == 'train':
                file_path = path.joinpath(file_path, strings[0], strings[1], f'{sample_id}.JPEG')
            else:
                file_path = path.joinpath(file_path, split, f'{sample_id}.JPEG')

            # label_name, *bboxes = row[1].strip().split(' ')
            label_name = row[1].strip().split(' ')[0]
            label_id = custom_dict[label_name]

            return ImageNetSplit.ClassificationImage.from_path(file_path, label_id)

        with open(path.joinpath(folder_path, 'LOC_val_solution.csv')) as csv_file, \
        h5py.File(h5py_path, 'w') as h5py_file:
            dt = csv.reader(csv_file)
            next(dt)  # skip header row

            dtype = np.dtype([
                ('width', np.uint16),
                ('height', np.uint16),
                ('image', h5py.vlen_dtype(ImageNetSplit.ClassificationImage.image_dtype)),
                ('label', ImageNetSplit.ClassificationImage.label_dtype),
            ])
            ds = h5py_file.create_dataset('data', (50000,), dtype=dtype)

            for i, row in enumerate(dt):  # read each row into a Sample
                print(f'\r{i}', end='')
                sup_img = _process_row(row)
                ds[i] = sup_img.to_disk()
                # break
            print()

            h5py_file.attrs['split'] = split
            h5py_file.attrs['size'] = i+1

        return cls(
            file_path=h5py_path,
            split=split,
            name='ImageNet',
        )

    def __getitem__(self, idx):
        sample = self.dataset['data'][idx]
        w, h = sample[0], sample[1]
        image = sample[2].reshape(h, w, 3)
        label = sample[3]

        return {
            'image': image,
            'label': label,
        }



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
