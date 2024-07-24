import json
from pathlib import Path
from typing import Literal, Any, TypeAlias

import cv2
import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QFont, QImage
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem

__all__ = [
    'RoiName',
    'PIXEL_CAL_FUNCTION',
    'compute_pixel_intensity',
    'RoiLabelObject',
    'PixVizResult',
]

RoiName: TypeAlias = str

PIXEL_CAL_FUNCTION = Literal['mean', 'median']


def compute_pixel_intensity(image: QImage | np.ndarray,
                            func: PIXEL_CAL_FUNCTION) -> float:
    """
    Compute the selected area pixel intensity

    :param image: image object, either ``PyQt6.QtGui.QImage`` or image ``numpy.array``
    :param func: ``PIXEL_CAL_FUNCTION`` {'mean', 'median'}
    :return:
    """
    if isinstance(image, QImage):
        import qimage2ndarray
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


class RoiLabelObject:
    rect_item: QGraphicsRectItem | None
    """set after selection"""
    text: QGraphicsTextItem | None
    """set after roi dialog"""
    background: QGraphicsRectItem | None
    """set after roi dialog"""
    func: PIXEL_CAL_FUNCTION
    """calculation func"""
    data: np.ndarray | None
    """(F,)"""

    def __init__(self):
        self.rect_item = None
        self.text = None
        self.name = None
        self.background = None
        self.func = 'mean'
        self.data = None

    def set_name(self, name: RoiName) -> None:
        """set name, text and background of the selected area

        :param name: roi name
        """
        self.name = name

        text_item = QGraphicsTextItem(name)
        text_item.setDefaultTextColor(QColor('white'))
        font = QFont()
        font.setPointSize(8)  # Adjust the size here
        font.setBold(True)
        text_item.setFont(font)
        text_item.setPos(self.rect_item.rect().topRight() + QPointF(5, 0))  # with space
        self.text = text_item
        assert self.name == self.text.toPlainText()

        # bg color
        text_rect = text_item.boundingRect()
        background_rect = QGraphicsRectItem(text_rect)
        background_color = QColor('green')
        background_color.setAlpha(128)
        background_rect.setBrush(background_color)
        background_rect.setPos(text_item.pos())
        self.background = background_rect

    def set_data(self, data: np.ndarray) -> None:
        """set calculated pixel intensity data"""
        self.data = data

    def to_meta(self, idx: int) -> dict[str, Any]:
        """to meta for saving"""
        return dict(name=self.name,
                    index=idx,
                    item=str(self.rect_item.rect()),
                    func=self.func)


# =========== #
# Load Result #
# =========== #

class PixVizResult:
    meta: dict[str, Any]
    dat: np.ndarray

    __slots__ = ('meta', 'dat')

    def __init__(self,
                 dat: Path | str,
                 meta: Path | str):
        """

        :param dat: .npy or .mat data path
        :param meta: .json data path
        """
        if Path(dat).suffix == '.npy':
            self.dat = np.load(dat)
        elif Path(dat).suffix == '.mat':
            raise NotImplementedError('')
        else:
            raise ValueError('')
        #
        with open(meta, 'r') as file:
            self.meta = json.load(file)

    @classmethod
    def load(cls,
             dat: Path | str,
             meta: Path | str):
        return cls(dat, meta)

    def __repr__(self):
        class_name = self.__class__.__name__
        roi_reprs = ", ".join(
            [f"{roi['name']} (index {roi['index']})" for roi in self.meta.values()]
        )
        return f"<{class_name}: [{roi_reprs}]>"

    __str__ = __repr__

    def get_index(self, name: RoiName) -> int:
        """
        Get `index` from `roi name`

        :param name: roi name
        :return: roi index
        """
        return self.meta[name]['index']

    def get_data(self, source: int | RoiName) -> np.ndarray:
        """
        Get roi data from either `index` or `name`

        :param source: if int type, get data from index;
            If string type, get data from roi name
        :return: data (F,)
        """
        if isinstance(source, int):
            pass
        elif isinstance(source, str):
            source = self.get_index(source)
        else:
            raise TypeError(f'invalid type: {type(source)}')

        return self.dat[source]

    def __getitem__(self, index: int) -> str:
        """
        Get `roi name` from index

        :param index: 0-based index
        :return: roi name
        """
        for name, roi in self.meta.items():
            if roi['index'] == index:
                return name

        raise IndexError(f'{index}')

    @property
    def n_rois(self) -> int:
        """number of roi selected"""
        return self.dat.shape[0]

    @property
    def n_frames(self) -> int:
        """number of frames (sequences)"""
        return self.dat.shape[1]
