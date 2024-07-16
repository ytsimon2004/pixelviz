import datetime
import pickle
import sys
import traceback
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QUrl, QRectF, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt6.QtGui import QTextCursor, QWheelEvent, QPen, QImage, QColor, QPixmap, QPainter
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsView,
    QGraphicsScene, QSlider, QTextEdit, QGraphicsRectItem, QSplitter, QDialog, QLineEdit, QRadioButton, QButtonGroup,
    QProgressBar
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core import PIXEL_CAL_FUNCTION
from pixviz.output import RoiType


def log_message(message: str) -> None:
    VideoLoaderApp.INSTANCE.log_message(message)


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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Settings")

        self.layout = QVBoxLayout()

        self.name_label = QLabel("Enter ROI name:")
        self.layout.addWidget(self.name_label)

        self.name_input = QLineEdit()
        self.layout.addWidget(self.name_input)

        self.method_label = QLabel("Select pixel calculation method:")
        self.layout.addWidget(self.method_label)

        self.mean_button = QRadioButton("Mean")
        self.median_button = QRadioButton("Median")

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.mean_button)
        self.button_group.addButton(self.median_button)

        self.layout.addWidget(self.mean_button)
        self.layout.addWidget(self.median_button)

        self.button_box = QHBoxLayout()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.button_box.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)
        self.setLayout(self.layout)

    def get_calculated_func(self) -> PIXEL_CAL_FUNCTION:
        if self.mean_button.isChecked():
            return 'mean'
        elif self.median_button.isChecked():
            return 'median'

    def get_roi_type(self) -> RoiType:
        return RoiType(self.name_input.text(), self.get_calculated_func())


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
        self.roi_rect_item = None
        self.rect: QRectF | None = None

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
            self.rect = QRectF(self.roi_start_pos, current_pos).normalized()
            self.roi_rect_item.setRect(self.rect)

    def mouseReleaseEvent(self, event):
        if self.drawing_roi:
            self.drawing_roi = False
            self.roi_start_pos = None
            self.roi_complete_signal.emit()

    def start_drawing_roi(self):
        self.drawing_roi = True
        self.roi_start_pos = None
        if self.roi_rect_item:
            self.scene().removeItem(self.roi_rect_item)
            self.roi_rect_item = None

    def process_frame(self, calculate_func: str = 'mean') -> None:
        if self.roi_rect_item:
            image = self.grab_frame()
            if image:
                roi_rect = self.roi_rect_item.rect().toRect()
                cropped_image = image.copy(roi_rect)

                if calculate_func == 'mean':
                    calc_pixel = self.calculate_average_pixel_value(cropped_image)
                elif calculate_func == 'median':
                    calc_pixel = self.calculate_median_pixel_value(cropped_image)
                else:
                    raise KeyboardInterrupt(f'unknown calculation method: {calculate_func}!')

                self.roi_average_signal.emit(calc_pixel)

    def grab_frame(self) -> QImage:
        # Grab the current frame from the video_item
        pixmap = QPixmap(self.video_item.boundingRect().size().toSize())
        painter = QPainter(pixmap)
        self.video_item.paint(painter, None, None)
        painter.end()
        return pixmap.toImage()

    @staticmethod
    def calculate_average_pixel_value(image: QImage):
        total_pixels = image.width() * image.height()
        if total_pixels == 0:
            return 0

        total_intensity = 0
        for x in range(image.width()):
            for y in range(image.height()):
                pixel = QColor(image.pixel(x, y))
                intensity = (pixel.red() + pixel.green() + pixel.blue()) / 3
                total_intensity += intensity

        return total_intensity / total_pixels

    @staticmethod
    def calculate_median_pixel_value(image: QImage):
        intensities = []
        for x in range(image.width()):
            for y in range(image.height()):
                pixel = QColor(image.pixel(x, y))
                intensity = (pixel.red() + pixel.green() + pixel.blue()) / 3
                intensities.append(intensity)

        if intensities:
            return np.median(intensities)
        return 0


class PlotView(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure())
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
    progress = pyqtSignal(int, float)
    """frame_number, result"""
    finished = pyqtSignal(list)
    """finish flag"""

    def __init__(self,
                 cap: cv2.VideoCapture,
                 roi_rect: QRectF,
                 calculate_func: PIXEL_CAL_FUNCTION):

        super().__init__()
        self.cap = cap
        self.roi_rect = roi_rect
        self.calculate_func = calculate_func

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        self.frame_results: np.ndarray | None = None

    def run(self):
        frame_values = np.full(self.total_frames, np.nan)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_number in range(self.total_frames):
            try:
                result = self.process_single_frame(frame_number)

                if result is None:
                    result = np.nan
                frame_values[frame_number] = result

                self.progress.emit(frame_number, result)
            except Exception as exc:
                print(f'Frame {frame_number} generated an exception: {exc}')
                traceback.print_exc()

        self.finished.emit(frame_values)

    def process_single_frame(self, frame_number):
        try:
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"Skipping frame {frame_number} due to read failure")
                return None

            roi_frame = frame[int(self.roi_rect.top()):int(self.roi_rect.bottom()),
                        int(self.roi_rect.left()):int(self.roi_rect.right())]

            if roi_frame.size == 0:
                print(f"Skipping frame {frame_number} due to empty ROI frame")
                return None

            frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

            if self.calculate_func == 'mean':
                return np.mean(frame)
            elif self.calculate_func == 'median':
                return np.median(frame)
        except Exception as e:
            print(f"Exception in processing frame {frame_number}: {e}")
            traceback.print_exc()
            return None


DEBUG_MODE = True


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

    plot_view: PlotView

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
        self.roi_button = QPushButton("Drag a ROI")
        control_group.addWidget(self.roi_button)

        # Result process
        self.process_button = QPushButton("Process")
        control_group.addWidget(self.process_button)

        self.process_progress = QProgressBar(self)
        self.process_progress.setRange(0, 100)
        self.process_progress.setValue(0)
        control_group.addWidget(self.process_progress)

        # Message log
        self.message_log = QTextEdit()
        self.message_log.setReadOnly(True)
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

        # media
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.positionChanged.connect(self.update_position)
        self.progress_bar.sliderMoved.connect(self.set_position)
        if DEBUG_MODE:
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
            self.log_message(f"Loaded Video: {file_path}")

            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

            self.log_message(f'total frames: {self.total_frames}, frame_rate: {self.frame_rate}')

            # Prompt for the sampling rate
            dialog = FrameRateDialog(default_value=self.frame_rate)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.frame_rate = dialog.get_sampling_rate()
                self.log_message(f"Sampling rate set to: {self.frame_rate} frames per second")
                self.timer.start(int(1000 // self.frame_rate))

    def load_result(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Pixviz Result File (*.pkl)')
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.log_message(f"Loaded Result: {file_path}")
            self.plot_view.load_from_file(file_path)

    def play_video(self):
        self.log_message("play")
        self.media_player.play()

    def pause_video(self):
        self.log_message("pause")
        self.media_player.pause()

    def _handle_media_status(self, status):
        match status:
            case QMediaPlayer.MediaStatus.EndOfMedia:
                self.log_message("End of media reached")
            case QMediaPlayer.MediaStatus.InvalidMedia:
                self.log_message("Invalid media")
            case QMediaPlayer.MediaStatus.NoMedia:
                self.log_message("No media loaded")
            case QMediaPlayer.MediaStatus.LoadingMedia:
                self.log_message("Loading media...")
            case QMediaPlayer.MediaStatus.LoadedMedia:
                self.log_message("Media loaded")
            case QMediaPlayer.MediaStatus.BufferedMedia:
                self.log_message("Media buffered")
            case QMediaPlayer.MediaStatus.StalledMedia:
                self.log_message("Media playback stalled")
            case _:
                self.log_message(f'unknown {status}')

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

    def keyPressEvent(self, event):
        current_position = self.media_player.position()
        frame_duration = 1000.0 / self.frame_rate  # duration of one frame in milliseconds

        match event.key():

            case Qt.Key.Key_Right:
                self.log_message("Right arrow key pressed")  # TODO bugfix receiver
                new_position = current_position + (10 * frame_duration)
                self.set_position(int(new_position))
            case Qt.Key.Key_Left:
                self.log_message("Left arrow key pressed")  # TODO bugfix receiver
                new_position = current_position - (10 * frame_duration)
                self.set_position(int(new_position))
            case Qt.Key.Key_Space:
                if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                    self.pause_video()
                else:
                    self.play_video()

    def process_frame(self):
        self.video_view.process_frame()

    def show_roi_settings_dialog(self):
        dialog = RoiSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            roi = dialog.get_roi_type()
            self.roi_list.append(roi)
            self.log_message(f"ROI name: {roi.name}, Calculation method: {roi.function}")

    # =========== #
    # Message Log #
    # =========== #

    def log_message(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.message_log.append(f"[{timestamp}] - {message}")
        self.message_log.moveCursor(QTextCursor.MoveOperation.End)

    def start_drawing_roi(self):
        self.video_view.start_drawing_roi()

    def process_all_frames(self):
        if len(self.roi_list) == 0:
            self.log_message("Please set an ROI first.")
            return

        self.plot_view.clear_plot()
        self.update_frame_number(0)

        self.frame_processor = FrameProcessor(self.cap, self.video_view.rect, self.roi_list[0].function)
        self.frame_processor.progress.connect(self.update_progress_and_frame)
        self.frame_processor.finished.connect(self.save_frame_values)
        self.frame_processor.start()

    @pyqtSlot(int, float)
    def update_progress_and_frame(self, frame_number: int, value: float):
        progress_value = int((frame_number / self.total_frames) * 100)
        self.process_progress.setValue(progress_value)
        pos = int((frame_number / self.total_frames) * self.media_player.duration())
        self.set_position(pos)
        self.update_frame_number(pos)

        if frame_number % 100 == 0:
            self.plot_view.update_plot(self.frame_processor.frame_results, start=0, end=frame_number + 1)

    @pyqtSlot(list)
    def save_frame_values(self, frame_values: np.ndarray) -> None:
        """save list[RoiType._asdict()] as pkl"""
        file = Path(self.video_path)
        output = file.with_stem(f'{file.stem}_pixviz').with_suffix('.pkl')

        out = []
        for roi in self.roi_list:
            res = RoiType(roi.name, roi.function, data=np.array(frame_values))
            out.append(res._asdict())

        with Path(output).open('wb') as file:
            pickle.dump(out, file)
            self.log_message("Averaged pixel values saved to averaged_pixel_values.npy")

    def main(self):
        self.show()
        self.setFocus()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoLoaderApp()
    window.main()
    sys.exit(app.exec())
