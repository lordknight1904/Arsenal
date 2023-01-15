# %%
import csv
from dataclasses import dataclass, fields
from functools import cached_property
import numpy as np
# import PIL
from PIL import Image
import h5py
from torch.utils.data import Dataset, default_collate
from typing import List, Type, ClassVar, Tuple


class BaseSplit(Dataset):
    '''
        A HDF5-based PyTorch Dataset
            - support for multiple workers (num_worker > 0)
            - support for asynchronous multiprocessing for pre-process dataset
    '''
    class IOMapping(dict):

        def __setitem__(self, key, value):
            # Remove any previous connections with these values
            if key in self:
                del self[key]
            if value in self:
                del self[value]
            dict.__setitem__(self, key, value)
            dict.__setitem__(self, value, key)

        def __delitem__(self, key):
            dict.__delitem__(self, self[key])
            dict.__delitem__(self, key)

        def __len__(self):
            """Returns the number of connections"""
            return dict.__len__(self) // 2

    @dataclass(slots=True)
    class InputImage:
        image: np.ndarray
        image_dtype: ClassVar[np.dtype] = np.uint8

        @classmethod
        def _img_from_path(cls, path:str) -> np.ndarray:
            return np.array(Image.open(path), dtype=cls.image_dtype)

        @classmethod
        def from_path(cls, path: str):
            return cls(cls._img_from_path(path))

        def to_disk(self) -> Tuple: 
            raise NotImplementedError

    def __init__(self, file_path, split, name):
        self.file_path = file_path
        self.split = split
        self.name = name

        with h5py.File(file_path, "r") as f:
            self.size = f.attrs['size']

    @cached_property
    def dataset(self): return h5py.File(self.file_path, 'r')

    def __len__(self): return self.size

    # def __getitem__(self, idx): return self.dataset['data'][idx]
    def __getitem__(self, idx): 
        raise NotImplementedError
    

if __name__ == '__main__':
    # img = InputImage.from_path('/home/leonard/data/ImageNet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00000036.JPEG')

    dataset_dir = '/home/leonard/data/ImageNet'
