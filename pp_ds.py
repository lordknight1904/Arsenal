# %%
import csv
import h5py

from typing import ClassVar, Tuple
from dataclasses import dataclass

import numpy as np
from pathlib import Path as path
from functools import partial

import autobot

@dataclass(slots=True)
class ClassificationImage(autobot.InputImage):
    label: np.uint
    label_dtype: ClassVar[np.dtype] = np.uint16

    @classmethod
    def from_path(cls, path: str, label: int):
        return cls(
            image=super(ClassificationImage, cls)._img_from_path(path, 384, 0.5, 0.5),
            # image=cls._read,
            label=cls.label_dtype(label)  # unit16 gives a maximum of 65535 classes
        )
    
    def to_disk(self) -> Tuple:
        return (self.image.shape[1], self.image.shape[0], self.image.flatten(), self.label,)

class ImageNetDataset(autobot.BaseSplit):

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
            return cls(h5py_path)

if __name__ == '__main__':
    import multiprocessing as mp
    import concurrent.futures

    folder_path = path.joinpath(path.home(), 'datasets', 'ImageNet')
    c2i, i2c = {}, {}
    h5py_path = path.joinpath(folder_path, f'val.h5')
    
    n_sample = 0
    with open(path.joinpath(folder_path, 'LOC_val_solution.csv')) as csv_file:
        dt = csv.reader(csv_file)
        next(dt)
        for _ in dt:
            n_sample += 1
    print('n_sample', n_sample)

    with open(path.joinpath(folder_path, 'LOC_synset_mapping.txt')) as f:
        for i, line in enumerate(f.readlines()):
            segments = line.strip().split(' ')
            label_idx = segments[0]
            c2i[label_idx] = i
            i2c[i] = label_idx
    
    def _get_gen(p):
        with open(p) as csv_file:
            dt = csv.reader(csv_file)
            next(dt)

            for row in dt:
                yield row

    def _read_row(row, _p, mapping):
        sample_id = row[0]

        strings = sample_id.split('_')
        split = strings[1] if strings[0] == 'ILSVRC2012' else 'train'
    
        file_path = path.joinpath(_p, 'ILSVRC', 'Data', 'CLS-LOC')
        if split == 'train':
            file_path = path.joinpath(file_path, strings[0], strings[1], f'{sample_id}.JPEG')
        else:
            file_path = path.joinpath(file_path, split, f'{sample_id}.JPEG')

        # label_name, *bboxes = row[1].strip().split(' ')
        label_name = row[1].strip().split(' ')[0]
        label_id = mapping[label_name]

        return ClassificationImage.from_path(file_path, label_id)
        # return (file_path, label_name)

    gen = _get_gen(path.joinpath(folder_path, 'LOC_val_solution.csv'))    

    with h5py.File(h5py_path, 'w') as h5py_file, \
        concurrent.futures.ProcessPoolExecutor(mp.cpu_count()) as executor:
        
        dtype = np.dtype([
            ('width', np.uint16),
            ('height', np.uint16),
            ('image', h5py.vlen_dtype(ClassificationImage.image_dtype)),
            ('label', ClassificationImage.label_dtype),
        ])
        ds = h5py_file.create_dataset('data', shape=(n_sample,), dtype=dtype)

        worker = partial(_read_row, _p=folder_path, mapping=c2i)
        for i, process_img in enumerate(executor.map(worker, gen, chunksize=10)):
            print(f'\r{i}', end='')
            ds[i] = process_img.to_disk()
            # break
        print()
        h5py_file.attrs['split'] = 'val'
        h5py_file.attrs['size'] = i+1
