from typing import NamedTuple

import numpy as np
from PyQt6.QtCore import QRectF
from typing_extensions import Self

from pixviz.core import PIXEL_CAL_FUNCTION

__all__ = ['RoiType']


class RoiType(NamedTuple):
    name: str
    """Roi name"""
    selection_area: QRectF
    """selected area"""
    function: PIXEL_CAL_FUNCTION
    """Roi calculated function"""
    data: np.ndarray | None = None
    """(F,)"""

    def with_data(self, data: np.ndarray) -> Self:
        return self._replace(data=data)
