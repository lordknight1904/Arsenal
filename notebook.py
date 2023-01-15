# %%
import autobot
import h5py
import matplotlib.pyplot as plt
from pathlib import Path as path

def main():
    imagenet_val_split = autobot.ImageNetSplit.from_path(
        folder_path=path.joinpath(path.home(), 'data', 'ImageNet'),
        split='val'
    )

if __name__ == '__main__':
    main()
    