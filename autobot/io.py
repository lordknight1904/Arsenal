# %%
import csv
import h5py
from dataclasses import dataclass, fields
from functools import cached_property
import numpy as np
# import PIL
from PIL import Image
from torch.utils.data import Dataset, default_collate
from typing import List, Type, ClassVar, Tuple


@dataclass(slots=True)
class InputImage:
    image: np.ndarray
    image_dtype: ClassVar[np.dtype]=np.float32

    @classmethod
    def _img_from_path(cls, path:str, size=0, mean=0., std=1.) -> np.ndarray:
        img = Image.open(path)
        img = img.resize((size, size))
        img = np.array(img) / 255
        img = (img - mean)/std

        img = np.array(img, dtype=cls.image_dtype)
        return img

    @classmethod
    def from_path(cls, path: str):
        return cls(cls._img_from_path(path))

    def to_disk(self) -> Tuple: 
        raise NotImplementedError


class BaseSplit(Dataset):
    '''
        A HDF5-based PyTorch Dataset
            - support for multiple workers (num_worker > 0)
            - support for asynchronous multiprocessing for pre-process dataset
    '''
    
    def __init__(self, file_path):
        self.file_path = file_path

        with h5py.File(file_path, "r") as f:
            self.size = f.attrs['size']

    @cached_property
    def dataset(self): return h5py.File(self.file_path, 'r')

    def __len__(self): return self.size

    def __getitem__(self, idx): 
        raise NotImplementedError
    

if __name__ == '__main__':
    # img = InputImage.from_path('/home/leonard/data/ImageNet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00000036.JPEG')

    dataset_dir = '/home/leonard/data/ImageNet'
