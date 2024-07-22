import datetime
import json
import re
import sys
import traceback
from pathlib import Path
from typing import ClassVar, Literal, Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QUrl, QRectF, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt6.QtGui import QTextCursor, QWheelEvent, QPen, QImage, QColor, QPixmap, QPainter, QKeyEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsView,
    QGraphicsScene, QSlider, QTextEdit, QGraphicsRectItem, QSplitter, QDialog, QLineEdit, QRadioButton, QButtonGroup,
    QProgressBar, QTableWidget, QTableWidgetItem
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pixviz.roi import RoiLabelObject, compute_pixel_intensity, PIXEL_CAL_FUNCTION

__all__ = ['run_gui']

LOGGING_TYPE = Literal['DEBUG', 'INFO', 'IO', 'WARNING', 'ERROR']
DEBUG_LOGGING = False


def log_message(message: str, log_type: LOGGING_TYPE = 'INFO', debug_mode: bool = DEBUG_LOGGING) -> None:
    VideoLoaderApp.INSTANCE.log_message(message, log_type, debug_mode=debug_mode)


class FrameRateDialog(QDialog):
    def __init__(self, default_value: float):
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

    def __init__(self, roi_object: RoiLabelObject, app: 'VideoLoaderApp'):
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

        self.selection_label = QLabel(f'Selected_area: {self.roi_object.rect_item.rect()}')
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
    """Signal to emit the roi_name and averaged pixel value"""

    roi_complete_signal = pyqtSignal(RoiLabelObject)
    """Signal to emit when ROI is completed"""

    def __init__(self):
        super().__init__()

        self.scale_factor: float = 1.0
        self.setScene(QGraphicsScene(self))
        self.video_item = QGraphicsVideoItem()
        self.scene().addItem(self.video_item)

        # ROI
        self.drawing_roi: bool = False  # isDrawing flag
        self.roi_start_pos = None
        self.current_roi_rect_item: QGraphicsRectItem | None = None
        self.roi_object: dict[str, RoiLabelObject] = {}

        self.pixmap_item = None

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

    def mouseMoveEvent(self, event):
        if self.drawing_roi and self.roi_start_pos:
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self.roi_start_pos, current_pos).normalized()
            self.current_roi_rect_item.setRect(rect)

    def mouseReleaseEvent(self, event):
        if self.drawing_roi:
            self.drawing_roi = False
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self.roi_start_pos, current_pos).normalized()
            self.current_roi_rect_item.setRect(rect)
            roi_object = RoiLabelObject()
            roi_object.rect_item = self.current_roi_rect_item
            self.current_roi_rect_item = None
            self.roi_complete_signal.emit(roi_object)

    def start_drawing_roi(self):
        self.drawing_roi = True
        self.roi_start_pos = None
        self.current_roi_rect_item = None

    def process_frame(self, calculate_func: PIXEL_CAL_FUNCTION = 'mean') -> None:
        if len(self.roi_object) != 0 and not self.drawing_roi:
            image = self.grab_frame()
            if image:
                signal = {}
                for name, roi in self.roi_object.items():
                    roi_rect = roi.rect_item.rect().toRect()
                    cropped_image = image.copy(roi_rect)
                    dat = compute_pixel_intensity(cropped_image, calculate_func)
                    signal[name] = dat

                self.roi_average_signal.emit(signal)

    def grab_frame(self) -> QImage:
        """Grab the current frame from the video_item"""
        pixmap = QPixmap(self.video_item.boundingRect().size().toSize())
        painter = QPainter(pixmap)
        self.video_item.paint(painter, None, None)
        painter.end()
        return pixmap.toImage()


class PlotView(QWidget):
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

        self._roi_lines: dict[str, Line2D] = {}

        self.ax = self.canvas.figure.subplots()
        self._set_axes()

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

    def add_axes(self, roi_name: str, **kwargs):
        """
        add axes for a given ROI name

        :param roi_name: roi name
        :param kwargs:
        :return:
        """
        self._roi_lines[roi_name] = self.ax.plot([], [], label=roi_name, **kwargs)[0]
        self.x_data[roi_name] = []
        self.y_data[roi_name] = []
        self.ax.legend()

    def delete_roi_line(self, roi_name: str):
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

    def update_plot(self,
                    frame_result: dict[str, np.ndarray],
                    start: int | None = None,
                    end: int | None = None):
        """

        :param frame_result: (R, F)
        :param start:
        :param end:
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

    def add_realtime_plot(self, values: dict[str, float]):
        """

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


class FrameProcessor(QThread):
    progress = pyqtSignal(int)
    """Frame Number"""
    results = pyqtSignal(dict)
    """Processed Result"""

    def __init__(self,
                 cap: cv2.VideoCapture,
                 rois: dict[str, RoiLabelObject],
                 view_size: tuple[int, int]):

        super().__init__()
        self.cap = cap
        self.rois = rois

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        self.view_size = view_size
        self.proc_results: dict[str, np.ndarray] = {
            name: np.full(self.total_frames, np.nan)
            for name in self.rois.keys()
        }

    @property
    def n_rois(self) -> int:
        return len(self.rois)

    def run(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_number in range(self.total_frames):

            try:
                result = self.process_single_frame()
                for name, val in result.items():
                    self.proc_results[name][frame_number] = val

                self.progress.emit(frame_number)

            except Exception as e:
                log_message(f'Frame {frame_number} generated an exception: {e}', log_type='ERROR')
                traceback.print_exc()

        self.results.emit(self.proc_results)

    def process_single_frame(self) -> dict[str, float]:
        _, frame = self.cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        origin_height, origin_width, *_ = frame.shape
        factor_width = origin_width / self.view_size[0]
        factor_height = origin_height / self.view_size[1]

        ret = {}
        for i, (name, roi) in enumerate(self.rois.items()):
            rect = roi.rect_item.rect()
            top = int(rect.top() * factor_height)
            bottom = int(rect.bottom() * factor_height)
            left = int(rect.left() * factor_width)
            right = int(rect.right() * factor_width)
            roi_frame = frame[top:bottom, left:right]

            ret[roi.name] = compute_pixel_intensity(roi_frame, roi.func)

        return ret


# ======== #
# Main App #
# ======== #

class VideoLoaderApp(QMainWindow):
    INSTANCE: ClassVar['VideoLoaderApp']

    load_video_button: QPushButton
    load_result_button: QPushButton

    video_view: VideoGraphicsView
    media_player: QMediaPlayer
    audio_output: QAudioOutput

    progress_bar: QSlider
    message_log: QTextEdit

    play_button: QPushButton
    pause_button: QPushButton

    roi_button: QPushButton
    delete_roi_button: QPushButton

    plot_view: PlotView

    roi_table: QTableWidget

    #
    frame_processor: FrameProcessor
    process_button: QPushButton
    process_progress: QProgressBar

    def __init__(self):
        super().__init__()

        VideoLoaderApp.INSTANCE = self

        # set after load
        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.total_frames: int | None = None
        self.frame_rate: float | None = None

        # container for roi_name:elements in QGraphicsVideoItem
        self.rois: dict[str, RoiLabelObject] = {}

        #
        self.setup_layout()
        self.setup_controller()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def setup_layout(self) -> None:
        self._set_dark_theme()

        # message log
        self.message_log = QTextEdit()
        self.message_log.setReadOnly(True)

        # windows
        self.setWindowTitle("Video Loader")
        self.setGeometry(100, 100, 1200, 800)

        # Create a central widget and set the layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Create the main left_splitter to hold message log and media components
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(left_splitter)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(right_splitter)

        # media
        self.video_view = VideoGraphicsView()
        left_splitter.addWidget(self.video_view)

        self.media_player = QMediaPlayer(self)
        self.video_view.set_media_player(self.media_player)

        # playing control
        media_control_layout = QHBoxLayout()
        media_control_widget = QWidget()
        media_control_widget.setLayout(media_control_layout)

        self.progress_bar = QSlider(Qt.Orientation.Horizontal)
        self.progress_bar.setRange(0, 100)
        media_control_layout.addWidget(self.progress_bar)

        self.play_button = QPushButton("Play")
        media_control_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        media_control_layout.addWidget(self.pause_button)
        left_splitter.addWidget(media_control_widget)

        # Plot View
        self.plot_view = PlotView()
        left_splitter.addWidget(self.plot_view)

        # ============== #
        # RIGHT SPLITTER #
        # ============== #

        # Table Widget for ROI Details
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(3)
        self.roi_table.setHorizontalHeaderLabels(["Name", "Selection", "Function"])
        right_splitter.addWidget(self.roi_table)

        # load button
        load_layout = QHBoxLayout()
        load_widget = QWidget()
        load_widget.setLayout(load_layout)
        right_splitter.addWidget(load_widget)

        self.load_video_button = QPushButton("Load Video")
        load_layout.addWidget(self.load_video_button)
        self.load_result_button = QPushButton("Load Result")
        load_layout.addWidget(self.load_result_button)

        # Control buttons layout
        control_group = QHBoxLayout()
        control_widget = QWidget()
        control_widget.setLayout(control_group)
        right_splitter.addWidget(control_widget)

        # ROI
        self.roi_button = QPushButton("Drag a Rect ROI")
        control_group.addWidget(self.roi_button)

        # Delete ROI button
        self.delete_roi_button = QPushButton("Delete ROI")
        control_group.addWidget(self.delete_roi_button)

        # Result process
        self.process_button = QPushButton("Process")
        control_group.addWidget(self.process_button)

        self.process_progress = QProgressBar(self)
        self.process_progress.setRange(0, 100)
        self.process_progress.setValue(0)
        control_group.addWidget(self.process_progress)

        right_splitter.addWidget(self.message_log)

    def _set_dark_theme(self) -> None:
        dark_stylesheet = """
        QWidget {
            background-color: #2E2E2E;
            color: white;
        }
        QGraphicsView {
            background-color: #2E2E2E;
            border: 1px solid #444;
        }
        QPushButton {
            background-color: #444;
            border: 1px solid #555;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #555;
        }
        QLineEdit, QTextEdit {
            background-color: #444;
            border: 1px solid #555;
            color: white;
        }
        QSlider::groove:horizontal {
            background: #444;
        }
        QSlider::handle:horizontal {
            background: #888;
            border: 1px solid #555;
            width: 10px;
        }
        QProgressBar {
            border: 1px solid #555;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #05B8CC;
        }
        QTableWidget::item {
            background-color: #2E2E2E;
            color: white;
        }
        QTableWidget::item:selected {
            background-color: #555;
            color: white;
        }
        """
        self.setStyleSheet(dark_stylesheet)
        plt.style.use('dark_background')

    def setup_controller(self) -> None:
        """controller for widgets"""
        # buttons
        self.load_video_button.clicked.connect(self.load_video)
        self.load_result_button.clicked.connect(self.load_result)
        self.roi_button.clicked.connect(self.start_drawing_roi)
        self.roi_button.clicked.connect(self.pause_video)

        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.process_button.clicked.connect(self.process_all_frames)
        self.delete_roi_button.clicked.connect(self.delete_selected_roi)

        # media
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.positionChanged.connect(self.update_position)
        self.progress_bar.sliderMoved.connect(self.set_position)
        self.media_player.mediaStatusChanged.connect(self._handle_media_status)

        # rois
        self.video_view.roi_complete_signal.connect(self.show_roi_settings_dialog)
        self.video_view.roi_average_signal.connect(self.plot_view.add_realtime_plot)

    def load_video(self) -> None:
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.log_message(f'Loaded Video: {file_path}', log_type='IO')

            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

            self.log_message(f'total frames: {self.total_frames}, frame_rate: {self.frame_rate}')

            # Prompt for the sampling rate
            dialog = FrameRateDialog(default_value=self.frame_rate)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.frame_rate = dialog.get_sampling_rate()
                self.log_message(f'Sampling rate set to: {self.frame_rate} frames per second')

            # show
            self.media_player.pause()
            self.media_player.setPosition(0)

    @property
    def video_item_size(self) -> tuple[int, int]:
        """
        Get ``QGraphicsVideoItem`` resized width and height

        :return: width and height
        """
        size = self.video_view.video_item.size()
        return size.width(), size.height()

    @property
    def data_output_file(self) -> Path:
        file = Path(self.video_path)
        return file.with_stem(f'{file.stem}_pixviz').with_suffix('.npy')

    @property
    def meta_output_file(self) -> Path:
        file = Path(self.video_path)
        return file.with_stem(f'{file.stem}_pixviz_meta').with_suffix('.json')

    def load_result(self) -> None:
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Pixviz Result File (*.npy)')
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.log_message(f'Loaded Result: {file_path}')
            self.load_from_file(file_path)

    def play_video(self) -> None:
        self.log_message("play", log_type='DEBUG')
        self.media_player.play()
        self.timer.start(int(1000 // self.frame_rate))

    def pause_video(self) -> None:
        self.log_message("pause", log_type='DEBUG')
        self.media_player.pause()
        self.timer.stop()

    def _handle_media_status(self, status) -> None:
        match status:
            case QMediaPlayer.MediaStatus.EndOfMedia:
                self.log_message("End of media reached", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.InvalidMedia:
                self.log_message("Invalid media", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.NoMedia:
                self.log_message("No media loaded", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.LoadingMedia:
                self.log_message("Loading media...", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.LoadedMedia:
                self.log_message("Media loaded", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.BufferedMedia:
                self.log_message("Media buffered", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.StalledMedia:
                self.log_message("Media playback stalled", log_type='DEBUG')
            case _:
                self.log_message(f'unknown {status}', log_type='DEBUG')

    def update_duration(self, duration: int) -> None:
        self.progress_bar.setRange(0, duration)

    def update_position(self, position: int) -> None:
        self.progress_bar.setValue(position)
        self.update_frame_number(position)

    def update_frame_number(self, position: int) -> None:
        frame_number = int((position / 1000.0) * self.frame_rate)
        self.video_view.frame_label.setText(f"Frame: {frame_number}")

    def set_position(self, position: int) -> None:
        self.media_player.setPosition(position)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """keyboard event handle"""
        if self.frame_rate is None:
            return

        current_position = self.media_player.position()
        frame_duration = 1000.0 / self.frame_rate  # ms

        match event.key():

            case Qt.Key.Key_Right:
                self.log_message('Right arrow key pressed')  # TODO bugfix receiver
                new_position = current_position + (10 * frame_duration)
                self.set_position(int(new_position))
            case Qt.Key.Key_Left:
                self.log_message('Left arrow key pressed')  # TODO bugfix receiver
                new_position = current_position - (10 * frame_duration)
                self.set_position(int(new_position))
            case Qt.Key.Key_Space:
                if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                    self.pause_video()
                else:
                    self.play_video()

    def process_frame(self) -> None:
        """realtime proc each frame"""
        self.video_view.process_frame()

    # ===================== #
    # ROI Setting and Table #
    # ===================== #

    def start_drawing_roi(self) -> None:
        self.log_message('Enable Drag mode')
        self.video_view.start_drawing_roi()

    def show_roi_settings_dialog(self, roi_object: RoiLabelObject) -> None:
        dialog = RoiSettingsDialog(roi_object, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.rois[roi_object.name] = roi_object
            self.video_view.roi_object[roi_object.name] = roi_object
            self.update_roi_table()

            roi_object.rect_item.setPen(QPen(QColor('green'), 2))
            self.video_view.scene().addItem(roi_object.background)
            self.video_view.scene().addItem(roi_object.text)

            self.plot_view.add_axes(roi_object.name)

    def update_roi_table(self) -> None:
        self.roi_table.setRowCount(len(self.rois))
        for row, (name, roi) in enumerate(self.rois.items()):
            self.roi_table.setItem(row, 0, QTableWidgetItem(name))
            self.roi_table.setItem(row, 1, QTableWidgetItem(str(roi.rect_item.rect())))
            self.roi_table.setItem(row, 2, QTableWidgetItem(roi.func))

    def delete_selected_roi(self) -> None:
        """delete the selected roi using the button click"""
        selected_items = self.roi_table.selectedItems()
        if not selected_items:
            self.log_message('No ROI selected for deletion.', log_type='ERROR')
            return

        selected_row = selected_items[0].row()
        roi_name = self.roi_table.item(selected_row, 0).text()
        self.roi_table.removeRow(selected_row)

        if roi_name in self.rois:
            roi_obj = self.rois.pop(roi_name)
            self.video_view.scene().removeItem(roi_obj.rect_item)
            self.video_view.scene().removeItem(roi_obj.text)
            self.video_view.scene().removeItem(roi_obj.background)

        #
        self.plot_view.delete_roi_line(roi_name)

        self.log_message(f"Deleted ROI: {roi_name}")

    # ================== #
    # Batch Process Mode #
    # ================== #

    def process_all_frames(self) -> None:
        if len(self.rois) == 0:
            self.log_message('Please set an ROI first.', log_type='ERROR')
            return

        self.plot_view.realtime_proc = False
        self.plot_view.clear_all()

        for name in self.rois.keys():
            self.plot_view.add_axes(name)

        self.update_frame_number(0)

        self.frame_processor = FrameProcessor(self.cap, self.rois, self.video_item_size)
        self.frame_processor.progress.connect(self.update_progress_and_frame)
        self.frame_processor.results.connect(self.save_frame_values)
        self.frame_processor.start()

    @pyqtSlot(int)
    def update_progress_and_frame(self, frame_number: int) -> None:
        """

        :param frame_number:
        :return:
        """
        progress_value = int((frame_number / self.total_frames) * 100)
        self.process_progress.setValue(progress_value)
        pos = int((frame_number / self.total_frames) * self.media_player.duration())
        self.set_position(pos)
        self.update_frame_number(pos)

        if frame_number % 100 == 0:
            self.plot_view.update_plot(self.frame_processor.proc_results, start=0, end=frame_number + 1)

    @pyqtSlot(dict)
    def save_frame_values(self, frame_values: dict[str, np.ndarray]) -> None:
        """
        Save meta info (`.json`) and (R, F) numpy ndarray (`.npy`)

        :param frame_values: name:result
        """
        if frame_values.keys() != self.rois.keys():
            self.log_message('roi name index incorrect', log_type='ERROR')

        self._save_meta()

        n_rois = len(frame_values)
        ret = np.zeros((n_rois, self.total_frames))
        for i, dat in enumerate(frame_values.values()):
            ret[i] = dat

        np.save(self.data_output_file, ret)
        self.log_message(f'Pixel intensity value saved to directory: {self.data_output_file.parent}', log_type='IO')

    def _save_meta(self):
        ret = {}
        for i, (name, roi) in enumerate(self.rois.items()):
            ret[name] = roi.to_meta(i)

        with self.meta_output_file.open('w') as f:
            json.dump(ret, f, sort_keys=True, indent=4)

    # ============= #
    # Result Reload #
    # ============= #

    def load_from_file(self, file: str) -> None:
        """Plot from result load"""
        file = Path(file)
        dat = np.load(file)
        meta_file = file.with_stem(f'{file.stem}_meta').with_suffix('.json')
        with open(meta_file, 'r') as file:
            meta = json.load(file)

        #
        view = self.plot_view
        self._reload(meta, dat)

        for name, line in self.plot_view._roi_lines.items():
            line.set_xdata(view.x_data[name])
            line.set_ydata(view.y_data[name])

        view.ax.relim()
        view.ax.autoscale_view()

        view.canvas.draw()

    def _reload(self, meta: dict[str, Any], dat: np.ndarray):

        # clear existing ROIs from the scene and plot
        self.plot_view.clear_all()
        self.rois.clear()
        self.video_view.roi_object.clear()

        self.roi_table.setRowCount(len(meta))
        for i, (name, it) in enumerate(meta.items()):
            self.roi_table.setItem(i, 0, QTableWidgetItem(name))
            self.roi_table.setItem(i, 1, QTableWidgetItem(it['item']))
            self.roi_table.setItem(i, 2, QTableWidgetItem(it['func']))

            self.plot_view.add_axes(name)
            d = dat[i]
            self.plot_view.x_data[name] = list(np.arange(len(d)))
            self.plot_view.y_data[name] = list(d)

            #
            roi_object = RoiLabelObject()
            rect_values = re.findall(r"[-+]?\d*\.\d+|\d+", it['item'])[1:]
            rect_values = list(map(float, rect_values))
            rect = QRectF(*rect_values)
            roi_object.rect_item = QGraphicsRectItem()
            roi_object.rect_item.setRect(rect)
            roi_object.rect_item.setPen(QPen(QColor('green'), 2))

            roi_object.set_name(name)
            roi_object.func = it['func']
            self.rois[name] = roi_object

            self.video_view.scene().addItem(roi_object.rect_item)
            self.video_view.scene().addItem(roi_object.background)
            self.video_view.scene().addItem(roi_object.text)
            self.video_view.roi_object[name] = roi_object

    # =========== #
    # Message Log #
    # =========== #

    def log_message(self, message: str, log_type: LOGGING_TYPE = 'INFO', debug_mode: bool = DEBUG_LOGGING) -> None:
        if not debug_mode and log_type == 'DEBUG':
            return

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        if self.message_log is None:
            print(message)
        else:
            color = self._get_log_type_color(log_type)
            log_entry = f'<span style="color:{color};">[{timestamp}] [{log_type}] - {message}</span><br>'

            self.message_log.insertHtml(log_entry)
            self.message_log.moveCursor(QTextCursor.MoveOperation.End)

    @staticmethod
    def _get_log_type_color(log_type: LOGGING_TYPE) -> str:
        match log_type:
            case 'INFO':
                return 'white'
            case 'IO':
                return 'cyan'
            case 'WARNING':
                return 'orange'
            case 'ERROR':
                return 'red'
            case _:
                return 'white'

    # ======= #

    def main(self):
        self.show()
        self.setFocus()


def run_gui():
    app = QApplication(sys.argv)
    window = VideoLoaderApp()
    window.main()
    sys.exit(app.exec())


if __name__ == '__main__':
    run_gui()
