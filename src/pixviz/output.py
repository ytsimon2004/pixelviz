from typing import NamedTuple

import numpy as np

from core import PIXEL_CAL_FUNCTION

__all__ = ['RoiType']


class RoiType(NamedTuple):
    name: str
    """Roi name"""
    function: PIXEL_CAL_FUNCTION
    """Roi calculated function"""
    data: np.ndarray | None = None
    """(F,)"""
