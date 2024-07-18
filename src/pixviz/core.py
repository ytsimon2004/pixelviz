from typing import Literal, Type

import cv2
import numpy as np
import qimage2ndarray
from PyQt6.QtGui import QImage

__all__ = ['PIXEL_CAL_FUNCTION',
           'compute_pixel_intensity']

PIXEL_CAL_FUNCTION: Type[str] = Literal['mean', 'median']


def compute_pixel_intensity(image: QImage | np.ndarray,
                            func: PIXEL_CAL_FUNCTION) -> float:
    """

    :param image:
    :param func:
    :return:
    """
    if isinstance(image, QImage):
        img = qimage2ndarray.rgb_view(image)

    elif isinstance(image, np.ndarray) and image.ndim == 3:  # RGB
        img = image
    else:
        raise TypeError('')

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if func == 'mean':
        return float(np.mean(img))
    elif func == 'median':
        return float(np.median(img))
