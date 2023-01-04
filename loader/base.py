# %%
from dataclasses import dataclass
import numpy as np
import PIL
from PIL import Image


@dataclass(slots=True)
class InputImage:
    image: np.ndarray

    @classmethod
    def from_path(cls, path: str):
        img = PIL.Image.open(path)
        return cls(np.array(img, dtype=np.uint8))

if __name__ == '__main__':
    img = InputImage.from_path('/home/leonard/arsenal/loader/ILSVRC2012_test_00000036.JPEG')
