# %%
import csv
import h5py
from dataclasses import dataclass, fields
from functools import cached_property
import numpy as np

import torch
from torch.utils.data import Dataset, default_collate

from typing import List, Type, ClassVar, Tuple, Union


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


@dataclass
class Output(Input): ...


@dataclass
class Batch:

    @classmethod
    def from_custom(cls, samples: List[Type[Union[Input, Output]]]):
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

    @classmethod
    def _read_row(cls, row) -> Type[Input]:
        raise NotImplementedError
    
    # @staticmethod
    # def to_disk(file_path):
    #     import concurrent.futures
        


class Trainer:

    def __init__(self,
        network: torch.nn.Module,
        train_ds: Type[BaseSplit],
        val_ds: Type[BaseSplit],
        test_ds: Type[BaseSplit],
        batch_size: int,
        batch_cls: Type[InputBatch],
    ):
        # self.device = torch.device('cuda:0')
        self.device = torch.device('cpu')
        self.batch_size = batch_size
        self.amp = torch.cuda.amp.autocast()

        self.network = network.to(self.device)

        self.train_loader = self._init_loader(train_ds) if train_ds else None
        self.val_loader = self._init_loader(val_ds, is_shuffle=False) if val_ds else None
        self.test_loader = self._init_loader(test_ds, is_shuffle=False) if test_ds else None

        self.writer = SummaryWriter()

        # self.loss_func = self.

        self.metrics = self._build_metrics()

        self.batch_cls = batch_cls

    def _collate_fn(self, samples: List[Type[autobot.Input]]):
        return self.batch_cls.from_supervise_samples(samples)

    def _init_loader(self, ds: Type[autobot.BaseSplit], is_shuffle: bool=True):
        return DataLoader(
            ds,
            batch_size=self.batch_size, shuffle=is_shuffle,
            collate_fn=self._collate_fn,
            num_workers=0,
        )
    
    # def _build_loss(self): ...

    def _build_metrics(self) -> List[Tuple[str, Type[metric.Metric]]]:
        return [
            ('Accuracy', metric.Accuracy(task="multiclass", num_classes=1000, top_k=1).to(self.device))
        ]

    def _step(self, batch):
        output = self.network(batch)

    def val_loop(self, step):
        for i, batch in enumerate(self.val_loader):
            print(f'\r{i:<3}/{len(self.val_loader):<3} {100*i/len(self.val_loader):.0f}%', end='')
            images, labels = batch['image'], batch['label']
            with torch.no_grad(), self.amp:
                preds = self.network(images.to(self.device))
            self.accuracy(preds, labels.to(self.device))

            for metric_name, metric in self.metrics:
                self.writer.add_scalar(f'{metric_name}/train', metric.compute(), step)
                metric.reset()

    def start(self,
        num_steps: int,
        from_checkpoint=None,
    ):
        if from_checkpoint == None: ...
        
        step = 0
        self.val_loop(step)


if __name__ == '__main__':
    # img = InputImage.from_path('/home/leonard/data/ImageNet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00000036.JPEG')

    dataset_dir = '/home/leonard/data/ImageNet'
