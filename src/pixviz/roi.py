import json
from pathlib import Path
from typing import Literal, Any, TypeAlias

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QFont, QImage
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem

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
                            func: PIXEL_CAL_FUNCTION,
                            debug_save: bool = False) -> float:
    """
    Compute the selected area pixel intensity

    :param image: image object, either ``PyQt6.QtGui.QImage`` or image ``numpy.array``
    :param func: ``PIXEL_CAL_FUNCTION`` {'mean', 'median'}
    :param debug_save: debug save cropped image
    :return:
    """
    if isinstance(image, QImage):
        import qimage2ndarray
        img = qimage2ndarray.rgb_view(image)
    elif isinstance(image, np.ndarray) and image.ndim == 3:  # RGB
        img = image
    else:
        raise TypeError('')

    if debug_save:
        plt.imshow(img, origin='upper')
        plt.show()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if func == 'mean':
        return float(np.mean(img))
    elif func == 'median':
        return float(np.median(img))


class RoiLabelObject:
    rect_item: QGraphicsRectItem
    """set after selection"""
    text: QGraphicsTextItem
    """set after roi dialog"""
    background: QGraphicsRectItem | None
    """set after roi dialog"""
    func: PIXEL_CAL_FUNCTION
    """calculation func"""
    data: np.ndarray | None
    """(F,)"""

    rotation_handle: QGraphicsEllipseItem
    angle: float

    def __init__(self):
        self.rect_item = QGraphicsRectItem()
        self.text = QGraphicsTextItem()
        self.name = None
        self.background = QGraphicsRectItem()
        self.func = 'mean'
        self.data = None

        # rotate
        self.angle = 0
        self.rotation_handle = QGraphicsEllipseItem(-5, -5, 10, 10)

    @property
    def rect_repr(self) -> str:
        """rect coordinates"""
        cord = self.rect_item.rect().getCoords()
        return str([round(c, 1) for c in cord])

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

        # Set rotation handle position
        handle_pos = self.rect_item.mapToScene(self.rect_item.rect().center())
        self.rotation_handle.setPos(handle_pos)
        self.rotation_handle.setBrush(QColor('red'))

    def set_data(self, data: np.ndarray) -> None:
        """set calculated pixel intensity data"""
        self.data = data

    def rotate(self, deg: float) -> None:
        self.angle += deg

    def update_rotation(self):
        self.angle %= 360
        if self.rect_item:
            center = self.rect_item.rect().center()
            transform = self.rect_item.transform()
            transform.reset()
            transform.translate(center.x(), center.y())
            transform.rotate(self.angle)
            transform.translate(-center.x(), -center.y())
            self.rect_item.setTransform(transform)

        self.update_element_position()

    def update_element_position(self):
        if self.text:
            self.text.setPos(self.rect_item.mapToScene(self.rect_item.rect().topRight()) + QPointF(5, 0))
        if self.background:
            self.background.setPos(self.text.pos())
        if self.rotation_handle:
            handle_pos = self.rect_item.mapToScene(self.rect_item.rect().center())
            self.rotation_handle.setPos(handle_pos)

    def to_meta(self, idx: int) -> dict[str, Any]:
        """to meta for saving"""
        return dict(name=self.name,
                    index=idx,
                    item=str(self.rect_item.rect()),
                    angle=self.angle,
                    func=self.func)

    def asdict(self) -> dict[str, Any]:
        return dict(
            name=self.name,
            rect=self.rect_item.rect(),
            angle=self.angle
        )


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
