import pickle
from pathlib import Path
from typing import NamedTuple
from typing_extensions import Self

import numpy as np

from core import PIXEL_CAL_FUNCTION

__all__ = ['RoiType',
           'load_roi_results']


class RoiType(NamedTuple):
    name: str
    """Roi name"""
    function: PIXEL_CAL_FUNCTION
    """Roi calculated function"""
    data: np.ndarray | None = None
    """(F,)"""

    def with_data(self, data: np.ndarray) -> Self:
        return self._replace(data=data)


def load_roi_results(file: Path | str) -> RoiType | list[RoiType]:
    with Path(file).open('rb') as f:
        res = pickle.load(f)

    if isinstance(res, dict):
        return RoiType(
            res['name'],
            res['function'],
            res['data']
        )
    elif isinstance(res, list):  # TODO multiple roi
        raise NotImplementedError('')
    else:
        raise TypeError('')
