# %%
import torch
import torch.cuda as cuda
from torch.utils.data import Dataset, DataLoader
import autobot

import torchmetrics as metric

import numpy as np

from dataclasses import dataclass
from typing import Type, List, Tuple
from torch.utils.tensorboard import SummaryWriter


@dataclass
class InputBatch:
    image: torch.Tensor
    label: torch.Tensor

    @classmethod
    def from_supervise_samples(cls, samples: List[Type[autobot.Input]]):
        return cls(
            image=torch.tensor(np.array([sample.image for sample in samples])),
            label=torch.tensor(np.array([sample.label for sample in samples])),
        )


class Trainer:

    def __init__(self,
        network: torch.nn.Module,
        train_ds: Type[autobot.BaseSplit],
        val_ds: Type[autobot.BaseSplit],
        test_ds: Type[autobot.BaseSplit],
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

        


