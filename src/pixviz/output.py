from typing import NamedTuple, Literal

import numpy as np

__all__ = ['RoiType']


class RoiType(NamedTuple):
    name: str
    """Roi name"""
    function: Literal['mean', 'median']
    """Roi calculated function"""
    data: np.ndarray | None = None
    """(F,)"""
