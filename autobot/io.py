# %%
import csv
import h5py
from dataclasses import dataclass, fields
from functools import cached_property
import numpy as np
from torch.utils.data import Dataset, default_collate
from typing import List, Type, ClassVar, Tuple


@dataclass(slots=True)
class Input:
    '''
        base class for h5py-based input
        read and write a row
    '''

    def to_disk(self) -> Tuple: 
        raise NotImplementedError

    @classmethod
    def from_disk(self, row):
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
