import datetime
import pickle
import sys
import traceback
from pathlib import Path
from typing import ClassVar, Literal

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

from pixviz.core import PIXEL_CAL_FUNCTION, compute_pixel_intensity
from pixviz.output import RoiType

__all__ = ['run_gui']

LOGGING_TYPE = Literal['DEBUG', 'INFO', 'IO', 'WARNING', 'ERROR']


def log_message(message: str, log_type: LOGGING_TYPE = 'INFO') -> None:
    VideoLoaderApp.INSTANCE.log_message(message, log_type)


class FrameRateDialog(QDialog):
    def __init__(self, default_value: float):
        super().__init__()
        self.setWindowTitle("Set Sampling Rate")

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

    def __init__(self, selection: QRectF):
        super().__init__()

        self.selection = selection
        self.setup_layout()
        self.setup_controller()

    def setup_layout(self) -> None:
        self.setWindowTitle("ROI Settings")
        self.layout = QVBoxLayout()
        self.name_label = QLabel("Enter ROI name:")
        self.layout.addWidget(self.name_label)

        self.name_input = QLineEdit()
        self.layout.addWidget(self.name_input)

        self.selection_label = QLabel(f'Selected_area: {self.selection}')
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

    def setup_controller(self) -> None:
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_calculated_func(self) -> PIXEL_CAL_FUNCTION:
        if self.mean_button.isChecked():
            return 'mean'
        elif self.median_button.isChecked():
            return 'median'

    def get_roi_type(self) -> RoiType:
        return RoiType(self.name_input.text(),
                       self.selection,
                       self.get_calculated_func())


class VideoGraphicsView(QGraphicsView):
    roi_average_signal = pyqtSignal(float)
    """Signal to emit the averaged pixel value"""

    roi_complete_signal = pyqtSignal()
    """Signal to emit when ROI is completed"""

    roi_start_signal = pyqtSignal()
    """Signal to emit when ROI drawing starts"""

    def __init__(self):
        super().__init__()

        self.scale_factor: float = 1.0
        self.setScene(QGraphicsScene(self))
        self.video_item = QGraphicsVideoItem()
        self.scene().addItem(self.video_item)

        # ROI
        self.drawing_roi: bool = False
        self.roi_start_pos = None
        self.roi_rect_item: QGraphicsRectItem | None = None
        self.rect_list: list[QRectF] = []

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

    def set_media_player(self, media_player: QMediaPlayer):
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
            self.roi_start_signal.emit()
            self.roi_start_pos = self.mapToScene(event.pos())
            if self.roi_rect_item is None:
                self.roi_rect_item = QGraphicsRectItem()
                self.roi_rect_item.setPen(QPen(Qt.GlobalColor.red, 2))
                self.scene().addItem(self.roi_rect_item)

    def mouseMoveEvent(self, event):
        if self.drawing_roi and self.roi_start_pos:
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self.roi_start_pos, current_pos).normalized()
            self.roi_rect_item.setRect(rect)

    def mouseReleaseEvent(self, event):
        if self.drawing_roi:
            self.drawing_roi = False
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self.roi_start_pos, current_pos).normalized()
            self.roi_rect_item.setRect(rect)
            self.rect_list.append(rect)  # Append the finalized rectangle
            self.roi_complete_signal.emit()

    def start_drawing_roi(self):
        self.drawing_roi = True
        self.roi_start_pos = None
        if self.roi_rect_item:
            self.scene().removeItem(self.roi_rect_item)
            self.roi_rect_item = None

    def process_frame(self, calculate_func: PIXEL_CAL_FUNCTION = 'mean') -> None:
        if self.roi_rect_item and not self.drawing_roi:
            image = self.grab_frame()
            if image:
                roi_rect = self.roi_rect_item.rect().toRect()
                cropped_image = image.copy(roi_rect)

                ret = compute_pixel_intensity(cropped_image, calculate_func)
                self.roi_average_signal.emit(ret)

    def grab_frame(self) -> QImage:
        # Grab the current frame from the video_item
        pixmap = QPixmap(self.video_item.boundingRect().size().toSize())
        painter = QPainter(pixmap)
        self.video_item.paint(painter, None, None)
        painter.end()
        return pixmap.toImage()


class PlotView(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.x_data = []
        self.y_data = []

        self.ax = self.canvas.figure.subplots()
        self.line, = self.ax.plot([], [])

        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Pixel Intensity')

    def update_plot(self, frame_result: np.ndarray,
                    start: int | None = None,
                    end: int | None = None):
        if frame_result is None:
            return

        if start is None:
            start = 0

        if end is None:
            end = len(frame_result)

        x_data = np.arange(start, end)
        y_data = frame_result[start:end]

        self.line.set_xdata(x_data)
        self.line.set_ydata(y_data)

        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()

    def clear_plot(self):
        self.x_data = []
        self.y_data = []
        self.line.set_xdata([])
        self.line.set_ydata([])

    def add_realtime_plot(self, value):
        self.x_data.append(len(self.x_data))
        self.y_data.append(value)

        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)

        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()

    def load_from_file(self, file: str) -> None:
        """TODO fix"""
        log_message(f'Plot view Load result from {file}')

        ret = []
        with Path(file).open('rb') as f:
            dat = pickle.load(f)

        for it in dat:
            ret.append(
                RoiType(
                    name=it['name'],
                    function=it['function'],
                    data=it['data']
                )
            )

        if len(ret) == 1:
            log_message(f'Set {ret[0].name} result from {file}')
            dat = ret[0].data
            self.line.set_xdata(np.arange(len(dat)))
            self.line.set_ydata(dat)

            self.ax.relim()
            self.ax.autoscale_view()

            self.canvas.draw()

        else:
            raise NotImplementedError('')


class FrameProcessor(QThread):
    progress = pyqtSignal(int)
    """Frame Number"""
    results = pyqtSignal(list)
    """Processed Result"""

    def __init__(self,
                 cap: cv2.VideoCapture,
                 roi_list: list[RoiType],
                 view_size: tuple[int, int]):

        super().__init__()
        self.cap = cap
        self.roi_list = roi_list

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        self.view_size = view_size
        self.frame_results: np.ndarray | None = None

    @property
    def n_rois(self) -> int:
        return len(self.roi_list)

    @property
    def selection_list(self) -> list[QRectF]:
        return [roi.selection_area for roi in self.roi_list]

    @property
    def calc_func_list(self) -> list[PIXEL_CAL_FUNCTION]:
        return [roi.function for roi in self.roi_list]

    def run(self):
        frame_values = np.full((self.n_rois, self.total_frames), np.nan)  # (R, F)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_number in range(self.total_frames):

            try:
                result = self.process_single_frame()  # (R,)
                frame_values[:, frame_number] = result
                self.progress.emit(frame_number)

            except Exception as e:
                log_message(f'Frame {frame_number} generated an exception: {e}', log_type='ERROR')
                traceback.print_exc()

        self.results.emit(frame_values)

    def process_single_frame(self) -> list[float]:
        _, frame = self.cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        origin_height, origin_width, *_ = frame.shape
        factor_width = origin_width / self.view_size[0]
        factor_height = origin_height / self.view_size[1]

        results = []
        for i, rect in enumerate(self.selection_list):
            top = int(rect.top() * factor_height)
            bottom = int(rect.bottom() * factor_height)
            left = int(rect.left() * factor_width)
            right = int(rect.right() * factor_width)
            roi_frame = frame[top:bottom, left:right]

            results.append(compute_pixel_intensity(roi_frame, self.calc_func_list[i]))

        return results


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
        self.roi_list: list[RoiType] = []

        #
        self.setup_layout()
        self.setup_controller()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def setup_layout(self):
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
        left_splitter.setStretchFactor(0, 3)  # Make video view take more space
        main_splitter.addWidget(left_splitter)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setStretchFactor(0, 3)  # Make video view take more space
        main_splitter.addWidget(right_splitter)

        # media
        self.video_view = VideoGraphicsView()
        left_splitter.addWidget(self.video_view)

        self.media_player = QMediaPlayer(self)
        self.video_view.set_media_player(self.media_player)

        # Progress Bar
        self.progress_bar = QSlider(Qt.Orientation.Horizontal)
        self.progress_bar.setRange(0, 100)
        progress_bar_layout = QVBoxLayout()
        progress_bar_layout.addWidget(self.progress_bar)

        # Plot View
        self.plot_view = PlotView()
        left_splitter.addWidget(self.plot_view)
        left_splitter.setStretchFactor(1, 1)

        left_splitter.addWidget(QWidget())
        left_splitter.widget(2).setLayout(progress_bar_layout)

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

        self.play_button = QPushButton("Play")
        control_group.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        control_group.addWidget(self.pause_button)

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

    def _set_dark_theme(self):
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

    def setup_controller(self):
        """controller for widgets"""
        # buttons
        self.load_video_button.clicked.connect(self.load_video)
        self.load_result_button.clicked.connect(self.load_result)
        self.roi_button.clicked.connect(self.start_drawing_roi)
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
        self.video_view.roi_start_signal.connect(self.pause_video)
        self.video_view.roi_complete_signal.connect(self.play_video)
        self.video_view.roi_average_signal.connect(self.plot_view.add_realtime_plot)

    def load_video(self):
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
                self.timer.start(int(1000 // self.frame_rate))

    @property
    def video_item_size(self) -> tuple[int, int]:
        """
        Get ``QGraphicsVideoItem`` resized width and height

        :return: width and height
        """
        size = self.video_view.video_item.size()
        return size.width(), size.height()

    def load_result(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Pixviz Result File (*.pkl)')
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.log_message(f'Loaded Result: {file_path}')
            self.plot_view.load_from_file(file_path)

    def play_video(self):
        self.log_message("play", log_type='DEBUG')
        self.media_player.play()

    def pause_video(self):
        self.log_message("pause", log_type='DEBUG')
        self.media_player.pause()

    def _handle_media_status(self, status):
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

    def update_duration(self, duration: int):
        self.progress_bar.setRange(0, duration)

    def update_position(self, position: int):
        self.progress_bar.setValue(position)
        self.update_frame_number(position)

    def update_frame_number(self, position: int):
        frame_number = int((position / 1000.0) * self.frame_rate)
        self.video_view.frame_label.setText(f"Frame: {frame_number}")

    def set_position(self, position: int):
        self.media_player.setPosition(position)

    def keyPressEvent(self, event: QKeyEvent):
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

    def process_frame(self):
        self.video_view.process_frame()

    # ===================== #
    # ROI Setting and Table #
    # ===================== #

    def start_drawing_roi(self):
        self.video_view.start_drawing_roi()

    def show_roi_settings_dialog(self) -> None:
        dialog = RoiSettingsDialog(self.video_view.rect_list[-1])  # TODO check, last one from append
        if dialog.exec() == QDialog.DialogCode.Accepted:
            roi = dialog.get_roi_type()

            if roi.name in [existed_roi.name for existed_roi in self.roi_list]:
                self.log_message(f'ROI name: {roi.name} already exist', log_type='ERROR')
                return

            self.roi_list.append(roi)

            self.log_message(f"ROI name: {roi.name}, Calculation method: {roi.function}")
            self.update_roi_table()

    def update_roi_table(self) -> None:
        self.roi_table.setRowCount(len(self.roi_list))
        for row, roi in enumerate(self.roi_list):
            self.roi_table.setItem(row, 0, QTableWidgetItem(roi.name))
            self.roi_table.setItem(row, 1, QTableWidgetItem(str(roi.selection_area)))
            self.roi_table.setItem(row, 2, QTableWidgetItem(roi.function))

    def delete_selected_roi(self):
        selected_items = self.roi_table.selectedItems()
        if not selected_items:
            self.log_message('No ROI selected for deletion.', log_type='ERROR')
            return

        selected_row = selected_items[0].row()
        roi_name = self.roi_table.item(selected_row, 0).text()

        self.roi_table.removeRow(selected_row)
        self.roi_list = [roi for roi in self.roi_list if roi.name != roi_name]

        self.log_message(f"Deleted ROI: {roi_name}")

    def process_all_frames(self):
        if len(self.roi_list) == 0:
            self.log_message('Please set an ROI first.', log_type='ERROR')
            return

        self.plot_view.clear_plot()
        self.update_frame_number(0)

        self.frame_processor = FrameProcessor(self.cap, self.roi_list, self.video_item_size)
        self.frame_processor.progress.connect(self.update_progress_and_frame)
        self.frame_processor.results.connect(self.save_frame_values)
        self.frame_processor.start()

    @pyqtSlot(int)
    def update_progress_and_frame(self, frame_number: int):
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
            self.plot_view.update_plot(self.frame_processor.frame_results, start=0, end=frame_number + 1)

    @pyqtSlot(list)
    def save_frame_values(self, frame_values: np.ndarray) -> None:
        """save list[RoiType._asdict()] as pkl

        :param frame_values: (R, F)
        """
        file = Path(self.video_path)
        output = file.with_stem(f'{file.stem}_pixviz').with_suffix('.pkl')

        dat = []
        for i, roi in enumerate(self.roi_list):
            res = RoiType(roi.name, roi.selection_area, roi.function, data=np.array(frame_values[i]))
            dat.append(res._asdict())

        with Path(output).open('wb') as f:
            pickle.dump(dat, f)
            self.log_message(f"Pixel intensity value saved to: {output}, with name: {[d['name'] for d in dat]}",
                             log_type='IO')

    # =========== #
    # Message Log #
    # =========== #

    def log_message(self, message: str, log_type: LOGGING_TYPE = 'INFO', debug_mode: bool = True) -> None:
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

    # === #
    # Run #
    # === #

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
