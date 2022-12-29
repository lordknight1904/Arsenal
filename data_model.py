from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class InputImage:
    pixels: np.ndarray
    dtype: np.dtype=np.uint8


# @dataclass
# class InputImageList:
