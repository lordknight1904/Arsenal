# %%
from dataclasses import dataclass, fields
import numpy as np
import PIL
from PIL import Image


@dataclass(slots=True)
class InputImage:
    image: np.ndarray

    def _validate_attr(self):
        if not isinstance(self.image, np.ndarray):
            print(f'Expect image of type ndarray but got {type(self.image)}')

        if (dim := len(self.image.shape)) != 3:
            print(f'Expect image with dimension of 3 but got {dim}')

    def __post_init__(self):
        self._validate_attr()

    @classmethod
    def from_path(cls, path: str):
        img = PIL.Image.open(path)
        return cls(np.array(img, dtype=np.uint8))

if __name__ == '__main__':
    img = InputImage.from_path('/home/leonard/arsenal/loader/ILSVRC2012_test_00000036.JPEG')
