import csv
import h5py
import typing
import torch

from torch import nn
from functools import partial
from dataclasses import dataclass
from typing import ClassVar, Tuple
import concurrent.futures

from pathlib import Path as path
import multiprocessing as mp
import numpy as np

from ..io import BaseSplit, InputImage


@dataclass(slots=True)
class ClassificationImage(InputImage):
    label: np.uint
    label_dtype: ClassVar[np.dtype] = np.uint16

    @classmethod
    def from_path(cls, path: str, label: int):
        return cls(
            image=super(ClassificationImage, cls)._img_from_path(path, 384, 0.5, 0.5),
            label=cls.label_dtype(label)  # unit16 gives a maximum of 65535 classes
        )
    
    def to_disk(self) -> Tuple:
        # return (self.image.shape[1], self.image.shape[0], self.image.flatten(), self.label,)
        return (self.image.flatten(), self.label,)


class ImageNetSplit(BaseSplit):
    
    def __init__(self, file_path):
        super().__init__(file_path)
        with h5py.File(file_path, "r") as f:
            self.w = f.attrs['width']
            self.h = f.attrs['height']
        print(self.size)

    def __getitem__(self, idx):
        # print('here')
        sample = self.dataset['data'][idx]
        image = sample[0].reshape(self.h, self.w, 3)
        label = sample[1]

        # return {
        #     'image': image,
        #     'label': label,
        # }

        return image, label

    @classmethod
    def _read_row(cls, row, _p, mapping):
        sample_id = row[0]

        strings = sample_id.split('_')
        # split = strings[1] if strings[0] == 'ILSVRC2012' else 'train'
        split = strings[1]
    
        file_path = path.joinpath(_p, 'ILSVRC', 'Data', 'CLS-LOC')
        if split == 'train':
            file_path = path.joinpath(file_path, strings[0], strings[1], f'{sample_id}.JPEG')
        else:
            file_path = path.joinpath(file_path, split, f'{sample_id}.JPEG')

        label_name = row[1].strip().split(' ')[0]
        label_id = mapping[label_name]

        # print(file_path, label_id)
        return ClassificationImage.from_path(file_path, label_id)

    @classmethod
    def from_path(cls, folder_path, csv_file):
        h5py_path = path.joinpath(folder_path, f'{csv_file.split(".csv")[0]}.h5')
        
        try:
            with h5py.File(h5py_path, 'r') as h5py_file:
                assert h5py_file.attrs.get('split', None) is not None
                assert h5py_file.attrs.get('size', None) is not None
        except FileNotFoundError:
            print(f'has not pre-processed. Start now.')
        except:
            print("Something else went wrong")
        else:
            print('reading pre-processed dataset')
            return cls(h5py_path)

        n_sample = 0
        with open(path.joinpath(folder_path, csv_file)) as c_file:
            dt = csv.reader(c_file)
            next(dt)
            for _ in dt:
                n_sample += 1
        print('n_sample', n_sample)

        #
        c2i, i2c = {}, {}
        with open(path.joinpath(folder_path, 'LOC_synset_mapping.txt')) as f:
            for i, line in enumerate(f.readlines()):
                segments = line.strip().split(' ')
                label_idx = segments[0]
                c2i[label_idx] = i
                i2c[i] = label_idx

        def _get_gen(p):
            with open(p) as c_file:
                dt = csv.reader(c_file)
                next(dt)

                for row in dt:
                    yield row

        gen = _get_gen(path.joinpath(folder_path, csv_file))

        with h5py.File(h5py_path, 'w') as h5py_file, \
            concurrent.futures.ProcessPoolExecutor(mp.cpu_count()) as executor:
            
            dtype = np.dtype([
                ('image', h5py.vlen_dtype(ClassificationImage.image_dtype)),
                ('label', ClassificationImage.label_dtype),
            ])
            ds = h5py_file.create_dataset('data', shape=(n_sample,), dtype=dtype)

            worker = partial(cls._read_row, _p=folder_path, mapping=c2i)
            for i, process_img in enumerate(executor.map(worker, gen, chunksize=10)):
                print(f'\r{i}', end='')
                ds[i] = process_img.to_disk()
                # break
            print()
            h5py_file.attrs['split'] = 'val'
            h5py_file.attrs['size'] = i+1

            h5py_file.attrs['width'] = 384
            h5py_file.attrs['height'] = 384


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
