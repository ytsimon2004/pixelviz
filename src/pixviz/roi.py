from typing import Literal, Any

import cv2
import numpy as np
import qimage2ndarray
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QFont, QImage
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem

__all__ = [
    'PIXEL_CAL_FUNCTION',
    'RoiLabelObject',
    'compute_pixel_intensity'
]

PIXEL_CAL_FUNCTION = Literal['mean', 'median']



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

    def __repr__(self):
        return f'RoiLabelObject: {self.name}'

    __str__ = __repr__

    def set_name(self, name: str) -> None:
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
        text_rect = text_item.boundingRect()  # Get the bounding rect of the text
        background_rect = QGraphicsRectItem(text_rect)
        background_color = QColor('green')
        background_color.setAlpha(128)
        background_rect.setBrush(background_color)
        background_rect.setPos(text_item.pos())
        self.background = background_rect

    def set_data(self, data: np.ndarray) -> None:
        self.data = data

    def to_meta(self, idx: int) -> dict[str, Any]:
        ret = {}
        ret['name'] = self.name
        ret['index'] = idx
        ret['item'] = str(self.rect_item.rect())
        ret['func'] = self.func

        return ret


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


def load_roi_results():
    pass
