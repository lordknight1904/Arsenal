import sys
import csv
import h5py
import typing
import torch

from torch import nn
from functools import partial
from dataclasses import dataclass
from typing import ClassVar, Tuple
import concurrent.futures

import multiprocessing as mp
import numpy as np
from pathlib import Path as path
from PIL import Image

from ..io import BaseSplit, Input


@dataclass(slots=True)
class InputImage(Input):
    image: np.ndarray
    image_dtype: ClassVar[np.dtype]=np.float32

    @classmethod
    def _img_from_path(cls, path:str, w=0, h=0, mean=0., std=1.) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        img = img.resize((w, h))
        img = np.array(img) / 255
        img = (img - mean)/std

        img = np.array(img, dtype=cls.image_dtype)
        return img

    @classmethod
    def from_path(cls, path: str):
        return cls(cls._img_from_path(path))


@dataclass(slots=True)
class ClassificationImage(InputImage):
    label: np.uint
    label_dtype: ClassVar[np.dtype] = np.int16  # int16 gives a maximum of 32.767 classes

    @classmethod
    def from_path(cls, path: str, label: int, w: int, h: int):
        return cls(
            image=super(ClassificationImage, cls)._img_from_path(path, w, h, 0.5, 0.5),
            label=cls.label_dtype(label)  
        )
    
    def to_disk(self) -> Tuple:
        # return (self.image.shape[1], self.image.shape[0], self.image.flatten(), self.label,)
        return (self.image.flatten(), self.label,)

    @classmethod
    def from_disk(cls, row, h, w) -> Tuple:
        return {
            'image': row[0].reshape(h, w, 3).transpose(2, 0, 1),
            'label': row[1],
        }


class ImageNetSplit(BaseSplit):
    
    def __init__(self, file_path):
        super().__init__(file_path)
        with h5py.File(file_path, "r") as f:
            self.h = f.attrs['height']
            self.w = f.attrs['width']

    def __getitem__(self, idx):
        return ClassificationImage.from_disk(self.dataset['data'][idx], self.h, self.w)

    @classmethod
    def _read_row(cls, row, _p, mapping, h=384, w=384):
        sample_id = row[0]

        file_path = path.joinpath(_p, 'ILSVRC', 'Data', 'CLS-LOC')
        if 'val' in sample_id:
            strings = sample_id.split('_')
            file_path = path.joinpath(file_path, strings[1], f'{sample_id}.JPEG')
        else:
            strings = sample_id.split('_')
            file_path = path.joinpath(file_path, 'train', strings[0], f'{sample_id}.JPEG')

        label_name = row[1].strip().split(' ')[0]
        label_id = mapping[label_name]

        # print(file_path, label_id)
        return ClassificationImage.from_path(file_path, label_id, w, h)

    @classmethod
    def from_path(cls, folder_path, csv_file, w=384, h=384, use_cache=True):
        h5py_path = path.joinpath(folder_path, f'{csv_file.split(".csv")[0]}_{w}_{h}.h5')
        
        if use_cache:
            with h5py.File(h5py_path, 'r') as h5py_file:
                assert h5py_file.attrs.get('split', None) is not None
                assert h5py_file.attrs.get('size', None) is not None
                assert h5py_file.attrs.get('h', None) is not None
                assert h5py_file.attrs.get('w', None) is not None
            return cls(h5py_path)

        n_sample = 0
        with open(path.joinpath(folder_path, csv_file)) as c_file:
            dt = csv.reader(c_file)
            next(dt)
            for _ in dt:
                n_sample += 1

        #
        c2i, i2c = {}, {}
        with open(path.joinpath(folder_path, 'LOC_synset_mapping.txt')) as f:
            for i, line in enumerate(f.readlines()):
                segments = line.strip().split(' ')
                label_idx = segments[0]
                c2i[label_idx] = i
                i2c[i] = label_idx

        with h5py.File(h5py_path, 'w') as h5py_file, \
            open(path.joinpath(folder_path, csv_file)) as c_file, \
            concurrent.futures.ProcessPoolExecutor(mp.cpu_count()) as executor:
            dt = csv.reader(c_file)
            next(dt)
            
            dtype = np.dtype([
                ('image', h5py.vlen_dtype(ClassificationImage.image_dtype)),
                ('label', ClassificationImage.label_dtype),
            ])
            ds = h5py_file.create_dataset(
                'data',
                dtype=dtype,
                shape=(1,),
                maxshape=(None,),
                # chunks=True,
                compression='gzip'
            )

            worker = partial(cls._read_row, _p=folder_path, mapping=c2i, w=w, h=h)
            for i, process_img in enumerate(
                executor.map(
                    worker,
                    dt,
                    chunksize=10
                )
            ):
                print(f'\r{i}', end='')
                ds.resize((i+1,))
                ds[i] = process_img.to_disk()
            print()

            # save meta data
            h5py_file.attrs['split'] = csv_file.split('_')[1]
            h5py_file.attrs['size'] = i+1

            h5py_file.attrs['height'] = h
            h5py_file.attrs['width'] = w
            
        return cls(h5py_path)


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
