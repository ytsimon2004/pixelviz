import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_gui import PixVizGUI

import cv2
import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt, QRectF, QThread, QLineF, QPointF
from PyQt6.QtGui import QWheelEvent, QPen
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QPushButton, QRadioButton,
    QButtonGroup, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QWidget, QGraphicsEllipseItem
)

from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pixviz.ui_logging import log_message

from pixviz.roi import RoiLabelObject, PIXEL_CAL_FUNCTION, compute_pixel_intensity, RoiName

__all__ = ['FrameRateDialog',
           'RoiSettingsDialog',
           'VideoGraphicsView',
           'PlotView',
           'FrameProcessor']


class FrameRateDialog(QDialog):
    """Frame rate check Dialog"""

    def __init__(self, default_value: float):
        """
        :param default_value: default frame rate
        """
        super().__init__()
        self.setWindowTitle("Set Sampling Rate")
        self._set_black_theme()

        self.layout = QVBoxLayout()

        self.label = QLabel("Enter the sampling rate (frames per second):")
        self.layout.addWidget(self.label)

        self.input = QLineEdit()
        self.input.setText(str(default_value))

        self.layout.addWidget(self.input)

        self.button_box = QHBoxLayout()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.button_box.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)
        self.setLayout(self.layout)

    def _set_black_theme(self) -> None:
        dark_style_theme = """
        QDialog {
            background-color: #2E2E2E;
            color: white;
        }
        QPushButton {
            background-color: #444;
            color: white;
            border: 1px solid #555;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #555;
        }
        QLineEdit, QLabel {
            background-color: #444;
            border: 1px solid #555;
            color: white;
        }
        """
        self.setStyleSheet(dark_style_theme)

    def get_sampling_rate(self) -> float:
        """sampling rate from text input"""
        return float(self.input.text())


class RoiSettingsDialog(QDialog):
    layout: QVBoxLayout
    name_label: QLabel
    name_input: QLineEdit
    selection_label: QLabel
    method_label: QLabel
    mean_button: QRadioButton
    median_button: QRadioButton
    button_group: QButtonGroup
    button_box: QHBoxLayout
    ok_button: QPushButton
    cancel_button: QPushButton

    def __init__(self, app: 'PixVizGUI',
                 roi_object: RoiLabelObject):
        """

        :param app: :class:`~pixviz.main_gui.PixVizGUI`
        :param roi_object: :class:`~pixviz.roi.RoiLabelObject`
        """
        super().__init__()

        self.roi_object = roi_object
        self.app = app
        self.setup_layout()
        self.setup_controller()

    def setup_layout(self) -> None:
        self.setWindowTitle("ROI Settings")
        self._set_dark_theme()
        self.layout = QVBoxLayout()
        self.name_label = QLabel("Enter ROI name:")
        self.layout.addWidget(self.name_label)

        self.name_input = QLineEdit()
        self.layout.addWidget(self.name_input)

        self.selection_label = QLabel(f'Selected_area: {self.roi_object.rect_repr}')
        self.layout.addWidget(self.selection_label)

        self.method_label = QLabel("Select pixel calculation method:")
        self.layout.addWidget(self.method_label)

        self.mean_button = QRadioButton("Mean")
        self.median_button = QRadioButton("Median")
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.mean_button)
        self.button_group.addButton(self.median_button)
        self.mean_button.setChecked(True)
        self.layout.addWidget(self.mean_button)
        self.layout.addWidget(self.median_button)

        self.button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.button_box.addWidget(self.ok_button)
        self.cancel_button = QPushButton("Cancel")
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)
        self.setLayout(self.layout)

    def _set_dark_theme(self) -> None:
        dark_stylesheet = """
        QDialog {
            background-color: #2E2E2E;
            color: white;
        }
        QPushButton {
            background-color: #444;
            color: white;
            border: 1px solid #555;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #555;
        }
        QLineEdit, QLabel, QRadioButton {
            background-color: #444;
            border: 1px solid #555;
            color: white;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def setup_controller(self) -> None:
        self.ok_button.clicked.connect(self._accept)
        self.cancel_button.clicked.connect(self._reject)
        self.name_input.textChanged.connect(self.edit)

    def edit(self, text: str) -> None:
        """check text input if a existed ROI name

        :param text: input text
        """
        if text in self.app.rois:
            self.name_input.setStyleSheet("QLineEdit {background-color: red;}")
            self.ok_button.setEnabled(False)
            log_message(f'{text} exists', log_type='ERROR')
        else:
            self.name_input.setStyleSheet("QLineEdit {background-color: black;}")
            self.ok_button.setEnabled(True)

    def get_calculated_func(self) -> PIXEL_CAL_FUNCTION:
        """get pixel intensity calculation function by button clicked"""
        if self.mean_button.isChecked():
            return 'mean'
        elif self.median_button.isChecked():
            return 'median'

    def _accept(self):
        """action of clicking `OK`"""
        self.roi_object.func = self.get_calculated_func()
        self.roi_object.set_name(self.name_input.text())
        log_message(f"ROI name: {self.roi_object.name}, Calculation method: {self.roi_object.func}")
        self.accept()

    def _reject(self):
        """action of clicking `Cancel`"""
        self.app.video_view.scene().removeItem(self.roi_object.rect_item)
        self.app.video_view.current_roi_rect_item = None
        self.app.video_view.drawing_roi = False
        self.reject()


class VideoGraphicsView(QGraphicsView):
    roi_average_signal = pyqtSignal(dict)
    """Signal to emit the roi_name and averaged pixel value in a single frame"""

    roi_complete_signal = pyqtSignal(RoiLabelObject)
    """Signal to emit when ROI selection is completed"""

    def __init__(self, app: 'PixVizGUI'):
        super().__init__()

        self.app = app
        self.scale_factor: float = 1.0
        self.setScene(QGraphicsScene(self))
        self.video_item = QGraphicsVideoItem()
        self.scene().addItem(self.video_item)

        # ROI
        self.drawing_roi: bool = False  # isDrawing flag
        self.moving_roi: bool = False
        self.rotating_roi: bool = False
        self.roi_start_pos: QPointF | None = None
        self.rotation_start_pos: QPointF | None = None
        self.move_start_pos: QPointF | None = None
        self.current_roi_rect_item: QGraphicsRectItem | None = None
        self.rois: dict[RoiName, RoiLabelObject] = {}  # set after roi dialog

        #
        self.media_player = None

        # Frame label
        self.frame_label = QLabel("Frame: 0")
        self.frame_label.setStyleSheet("color: red; font-weight: bold;")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.frame_label.setMargin(10)
        self.frame_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.frame_label_layout = QVBoxLayout()
        self.frame_label_layout.addWidget(self.frame_label)
        self.frame_label_layout.addStretch()
        self.frame_label_widget = QWidget()
        self.frame_label_widget.setLayout(self.frame_label_layout)
        self.frame_label_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.scene().addWidget(self.frame_label_widget)

        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_media_player(self, media_player: QMediaPlayer) -> None:
        media_player.setVideoOutput(self.video_item)
        self.media_player = media_player

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self) -> None:
        self.scale_factor *= 1.1
        self.apply_zoom()

    def zoom_out(self) -> None:
        self.scale_factor /= 1.1
        self.apply_zoom()

    def apply_zoom(self) -> None:
        self.resetTransform()
        self.scale(self.scale_factor, self.scale_factor)

    def mousePressEvent(self, event):
        if self.drawing_roi:
            self.roi_start_pos = self.mapToScene(event.pos())
            if self.current_roi_rect_item is None:
                self.current_roi_rect_item = QGraphicsRectItem()
                self.current_roi_rect_item.setPen(QPen(Qt.GlobalColor.red, 2))
                self.scene().addItem(self.current_roi_rect_item)
        else:
            # roi rotation and moving mode
            item = self.itemAt(event.pos())
            if isinstance(item, QGraphicsEllipseItem):
                self.rotating_roi = True
                self.rotation_start_pos = self.mapToScene(event.pos())
                self.current_roi_rect_item = next(
                    (roi.rect_item for roi in self.rois.values() if roi.rotation_handle == item), None)
            elif isinstance(item, QGraphicsRectItem) and item in [roi.rect_item for roi in self.rois.values()]:
                self.moving_roi = True
                self.move_start_pos = self.mapToScene(event.pos())
                self.current_roi_rect_item = item
            else:
                super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """mouse movement for `drag`, `rotate`, `move` modes"""
        if self.current_roi_rect_item:
            roi_object = next((
                roi for roi in self.rois.values()
                if roi.rect_item == self.current_roi_rect_item
            ), None)
        else:
            roi_object = None

        match (self.drawing_roi, self.rotating_roi, self.moving_roi):
            case (True, _, _) if self.roi_start_pos is not None:
                current_pos = self.mapToScene(event.pos())
                rect = QRectF(self.roi_start_pos, current_pos).normalized()
                self.current_roi_rect_item.setRect(rect)
            case (_, True, _) if roi_object is not None:
                current_pos = self.mapToScene(event.pos())
                center = self.current_roi_rect_item.rect().center()
                initial_line = QLineF(center, self.rotation_start_pos)
                current_line = QLineF(center, current_pos)
                angle = current_line.angleTo(initial_line)
                roi_object.rotate(angle)
                self.rotation_start_pos = current_pos
                roi_object.update_rotation()
                self.app.update_roi_table()
            case (_, _, True) if roi_object is not None:
                current_pos = self.mapToScene(event.pos())
                delta = current_pos - self.move_start_pos
                self.move_start_pos = current_pos
                self.move_roi_rect(roi_object, delta)
                roi_object.update_rotation()
                roi_object.update_element_position()
                self.app.update_roi_table()
            case _:
                super().mouseMoveEvent(event)

    @staticmethod
    def move_roi_rect(roi_object, delta):
        rect = roi_object.rect_item.rect()
        new_rect = rect.translated(delta)
        roi_object.rect_item.setRect(new_rect)

    def mouseReleaseEvent(self, event):
        if self.drawing_roi:
            self.drawing_roi = False
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self.roi_start_pos, current_pos).normalized()
            self.current_roi_rect_item.setRect(rect)
            roi_object = RoiLabelObject()
            roi_object.rect_item = self.current_roi_rect_item
            roi_object.update_element_position()  # Ensure correct position update
            self.scene().addItem(roi_object.rotation_handle)
            self.rois[roi_object.name] = roi_object
            self.roi_complete_signal.emit(roi_object)
        elif self.rotating_roi:
            self.rotating_roi = False
        elif self.moving_roi:
            self.moving_roi = False
        else:
            super().mouseReleaseEvent(event)

    def start_drawing_roi(self):
        self.drawing_roi = True
        self.roi_start_pos = None
        self.current_roi_rect_item = None

    def process_frame(self) -> None:
        if len(self.rois) != 0 and not self.drawing_roi:
            signal = process_single_frame(self.rois, self.app.cap, self.app.video_item_size)
            if signal is not None:
                self.roi_average_signal.emit(signal)


class PlotView(QWidget):
    """mpl plot view"""
    clear_button: QPushButton
    canvas: FigureCanvas

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setup_layout()
        self.setup_controller()

        # for realtime plot
        self.realtime_proc: bool = True
        self.x_data = {}
        self.y_data = {}

        self._roi_lines: dict[RoiName, Line2D] = {}

        self.ax = self.canvas.figure.subplots()
        self._set_axes()

        # reload
        self.enable_axvline: bool = False
        self.vertical_line: Line2D | None = None

    def setup_layout(self):
        layout = QVBoxLayout(self)
        toolbar_layout = QHBoxLayout()
        self.canvas = FigureCanvas(Figure())
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar_layout.addWidget(toolbar)

        self.clear_button = QPushButton("Clear")
        toolbar_layout.addWidget(self.clear_button)
        layout.addLayout(toolbar_layout)
        layout.addWidget(self.canvas)

    def _set_axes(self):
        for s in ('top', 'right', 'bottom'):
            self.ax.spines[s].set_visible(False)

        self.ax.get_xaxis().set_visible(False)
        self.ax.set_ylabel('Pixel Intensity')

    def setup_controller(self):
        self.clear_button.clicked.connect(self.clear_axes)

    def add_axes(self, roi_name: RoiName, **kwargs):
        """
        Add axes for a given ROI name

        :param roi_name: roi name
        :param kwargs: additional arguments to ``ax.plot()``
        :return:
        """
        self._roi_lines[roi_name] = self.ax.plot([], [], label=roi_name, **kwargs)[0]
        self.x_data[roi_name] = []
        self.y_data[roi_name] = []
        self.ax.legend()

    def delete_roi_line(self, roi_name: RoiName):
        """
        Remove line, legend, and

        :param roi_name:
        :return:
        """
        # remove line and legend
        self._roi_lines[roi_name].remove()
        self.ax.legend()

        try:
            del self._roi_lines[roi_name]
            self.x_data[roi_name] = []
            self.y_data[roi_name] = []
        except KeyError:
            log_message(f'{roi_name} not exist', log_type='ERROR')

    def clear_all(self):
        """clear all elements in the plot view"""
        self.x_data = {}
        self.y_data = {}

        for name, line in list(self._roi_lines.items()):
            line.remove()
            del self._roi_lines[name]

    def clear_axes(self):
        """clear axes without removing data"""
        self.ax.cla()

        # add back axes for rendering
        for name in self._roi_lines:
            self.add_axes(name)

    def update_batch_plot(self,
                          frame_result: dict[RoiName, np.ndarray],
                          start: int | None = None,
                          end: int | None = None):
        """
        Process batch mode update

        :param frame_result: roi:
        :param start: starting frame number, if None then 0.
        :param end: ending frame number, if None then all frames
        :return:
        """
        if frame_result is None:
            return

        if start is None:
            start = 0

        if end is None:
            end = len(frame_result)

        x_data = np.arange(start, end)

        for name, res in frame_result.items():
            y_data = res[start:end]
            self._roi_lines[name].set_xdata(x_data)
            self._roi_lines[name].set_ydata(y_data)

        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()

    def update_realtime_plot(self, values: dict[RoiName, float]):
        """
        Realtime processed update

        :param values: roi name: value
        :return:
        """
        if self.realtime_proc:

            for name, val in values.items():

                if name not in self.x_data or name not in self.y_data:
                    continue

                self.x_data[name].append(len(self.x_data[name]))
                self.y_data[name].append(val)

            for name, line in self._roi_lines.items():
                line.set_xdata(self.x_data[name])
                line.set_ydata(self.y_data[name])

            self.ax.relim()
            self.ax.autoscale_view()

            self.canvas.draw()

    def set_axvline(self):
        self.vertical_line = self.ax.axvline(x=0, color='pink', linestyle='--', zorder=1)

    def update_vertical_line_position(self, frame_number: int):
        """Update the vertical line position based on the current frame number."""
        self.vertical_line.set_xdata([frame_number])
        self.canvas.draw()


class FrameProcessor(QThread):
    progress = pyqtSignal(int)
    """Frame Number"""
    results = pyqtSignal(dict)
    """Processed Result, dict[RoiName, np.ndarray]"""

    def __init__(self,
                 cap: cv2.VideoCapture,
                 rois: dict[RoiName, RoiLabelObject],
                 view_size: tuple[int, int]):
        """

        :param app: :class:`~pixviz.main_gui.PixVizGUI`
        :param cap: ``cv2.VideoCapture``
        :param rois: dict of [roi_name, :class:`~pixviz.roi.RoiLabelObject`]
        :param view_size: rescaled view size
        """

        super().__init__()
        self.cap = cap
        self.rois = rois

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        self.view_size = view_size
        self.proc_results: dict[RoiName, np.ndarray] = {
            name: np.full(self.total_frames, np.nan)
            for name in self.rois.keys()
        }

    def run(self):
        """QThread run"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_number in range(self.total_frames):

            try:
                result = process_single_frame(self.rois, self.cap, self.view_size)
                for name, val in result.items():
                    self.proc_results[name][frame_number] = val

                self.progress.emit(frame_number)

            except Exception as e:
                log_message(f'Frame {frame_number} generated an exception: {e}', log_type='ERROR')
                traceback.print_exc()

        self.results.emit(self.proc_results)


def process_single_frame(roi_dict: dict[RoiName, RoiLabelObject],
                         cap: cv2.VideoCapture,
                         video_item_size: tuple[int, int]) -> dict[RoiName, float] | None:
    """
    single frame calculation (used for batch run)

    :param roi_dict: dict of ``RoiName``:``RoiLabelObject``
    :param cap: video capture
    :param video_item_size:
    :return: dict of name:processed_results
    """
    ret, frame = cap.read()

    if not ret:
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    origin_height, origin_width, *_ = frame.shape
    factor_width = origin_width / video_item_size[0]
    factor_height = origin_height / video_item_size[1]

    sig = {}
    for i, (name, roi) in enumerate(roi_dict.items()):

        rect = roi.rect_item.rect()
        top = int(rect.top() * factor_height)
        bottom = int(rect.bottom() * factor_height)
        left = int(rect.left() * factor_width)
        right = int(rect.right() * factor_width)

        if roi.angle != 0:
            center = (int((left + right) / 2), int((top + bottom) / 2))
            rotation_matrix = cv2.getRotationMatrix2D(center, roi.angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (origin_width, origin_height))
            roi_frame = rotated_frame[top:bottom, left:right]
        else:
            roi_frame = frame[top:bottom, left:right]

        sig[roi.name] = compute_pixel_intensity(roi_frame, roi.func)

    return sig
