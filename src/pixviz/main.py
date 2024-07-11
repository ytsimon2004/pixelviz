import sys
from typing import Literal

import numpy as np
from PyQt6.QtCore import Qt, QUrl, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import QTextCursor, QWheelEvent, QPen, QImage, QColor, QPixmap, QPainter
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsView,
    QGraphicsScene, QSlider, QTextEdit, QGraphicsRectItem, QSplitter, QDialog, QLineEdit, QRadioButton, QButtonGroup
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pixviz.output import RoiType


class SamplingRateDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Sampling Rate")

        self.layout = QVBoxLayout()

        self.label = QLabel("Enter the sampling rate (frames per second):")
        self.layout.addWidget(self.label)

        self.input = QLineEdit()
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

    def get_calculated_func(self) -> Literal['mean', 'median']:
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

        self.pixmap_item = None

        #
        self.media_player = None

    def set_media_player(self, media_player: QMediaPlayer):
        media_player.setVideoOutput(self.video_item)
        self.media_player = media_player
        # self.media_player.positionChanged.connect(self.process_frame)

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
            self.roi_start_pos = None
            self.roi_complete_signal.emit()

    def start_drawing_roi(self):
        self.drawing_roi = True
        self.roi_start_pos = None
        if self.roi_rect_item:
            self.scene().removeItem(self.roi_rect_item)
            self.roi_rect_item = None

    def process_frame(self, calculate_func: str = 'mean'):
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

    def grab_frame(self):
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure())
        self.layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.subplots()
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot(self.x_data, self.y_data)

        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Average Pixel Intensity')

    def update_plot(self, value):
        self.x_data.append(len(self.x_data))
        self.y_data.append(value)

        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)

        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()


class VideoLoaderApp(QMainWindow):
    layout: QHBoxLayout
    main_splitter: QSplitter
    splitter: QSplitter

    progress_bar_layout: QVBoxLayout

    load_button: QPushButton

    video_view: VideoGraphicsView
    media_player: QMediaPlayer
    control_layout: QHBoxLayout
    audio_output: QAudioOutput

    progress_bar: QSlider
    frame_label: QLabel
    message_log: QTextEdit

    play_button: QPushButton
    pause_button: QPushButton

    control_panel: QHBoxLayout
    roi_button: QPushButton

    plot_view: PlotView

    def __init__(self):
        super().__init__()

        #
        self.sampling_rate: float | None = None
        self.roi: RoiType | None = None

        #
        self.setup_windows()

        #
        self.button_load_video()
        self.setup_media_player()
        self.setup_audio()

        #
        self.setup_progress_bar()
        self.setup_frame_label()
        self.setup_message_log()

        #
        self.control_media_player()

        #
        self.setup_control_panel()

        #
        self.setup_plot_view()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def setup_windows(self):
        self.setWindowTitle("Video Loader")
        self.setGeometry(100, 100, 1200, 800)

        # Create a central widget and set the layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # self.layout = QVBoxLayout()
        self.layout = QHBoxLayout()
        central_widget.setLayout(self.layout)

        # Create the main splitter to hold message log and media components
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.layout.addWidget(self.main_splitter)

    def button_load_video(self):
        """Create a button to load the video file"""
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button, alignment=Qt.AlignmentFlag.AlignTop)

    def load_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.log_message(f"Loaded Video: {file_path}")

            # Prompt for the sampling rate
            dialog = SamplingRateDialog()
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.sampling_rate = dialog.get_sampling_rate()
                self.log_message(f"Sampling rate set to: {self.sampling_rate} frames per second")
                self.timer.start(int(1000 // self.sampling_rate))

    # ========== #
    # Video View #
    # ========== #

    def setup_media_player(self):
        """Video view for playing the video"""
        self.video_view = VideoGraphicsView()
        self.video_view.roi_complete_signal.connect(self.show_roi_settings_dialog)
        self.video_view.roi_start_signal.connect(self.pause_video)
        self.video_view.roi_complete_signal.connect(self.play_video)

        # Use QSplitter to allow resizing
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.video_view)
        self.splitter.setStretchFactor(0, 3)  # Make video view take more space
        self.layout.addWidget(self.splitter)

        self.media_player = QMediaPlayer(self)
        self.video_view.set_media_player(self.media_player)

        self.video_view.roi_average_signal.connect(self.update_plot)

        # Add the splitter to the main splitter
        self.main_splitter.addWidget(self.splitter)

    def control_media_player(self):
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.mediaStatusChanged.connect(self.handle_media_status)

        # Control buttons layout
        self.control_layout = QHBoxLayout()
        self.layout.addLayout(self.control_layout)

        # Play button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        self.control_layout.addWidget(self.play_button)

        # Pause button
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        self.control_layout.addWidget(self.pause_button)

    def play_video(self):
        self.log_message("play")
        self.media_player.play()

    def pause_video(self):
        self.log_message("pause")
        self.media_player.pause()

    def handle_media_status(self, status):
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

    # ============ #
    # Audio output #
    # ============ #

    def setup_audio(self):
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)

    # ============ #
    # Progress Bar #
    # ============ #

    def setup_progress_bar(self):
        self.progress_bar = QSlider(Qt.Orientation.Horizontal)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.sliderMoved.connect(self.set_position)

        self.progress_bar_layout = QVBoxLayout()
        self.progress_bar_layout.addWidget(self.progress_bar)

    def setup_frame_label(self):
        self.frame_label = QLabel("Frame: 0")
        self.layout.addWidget(self.frame_label, alignment=Qt.AlignmentFlag.AlignTop)

    def update_duration(self, duration):
        self.progress_bar.setRange(0, duration)

    def update_position(self, position):
        self.progress_bar.setValue(position)
        self.update_frame_number(position)

    def update_frame_number(self, position):
        frame_number = int((position / 1000.0) * self.sampling_rate)
        self.frame_label.setText(f"Frame: {frame_number}")

    def set_position(self, position):
        self.media_player.setPosition(position)

    def keyPressEvent(self, event):
        current_position = self.media_player.position()
        frame_duration = 1000.0 / self.sampling_rate  # duration of one frame in milliseconds

        if event.key() == Qt.Key.Key_Right:
            self.log_message("Right arrow key pressed")
            new_position = current_position + (10 * frame_duration)
            self.media_player.setPosition(int(new_position))
        elif event.key() == Qt.Key.Key_Left:
            self.log_message("Left arrow key pressed")
            new_position = current_position - (10 * frame_duration)
            self.media_player.setPosition(int(new_position))

    def process_frame(self):
        self.video_view.process_frame()

    def show_roi_settings_dialog(self):
        dialog = RoiSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.roi = dialog.get_roi_type()
            self.log_message(f"ROI name: {self.roi.name}, Calculation method: {self.roi.function}")

    # =========== #
    # Message Log #
    # =========== #

    def setup_message_log(self) -> None:
        self.message_log = QTextEdit()
        self.message_log.setReadOnly(True)
        self.main_splitter.addWidget(self.message_log)
        self.main_splitter.setStretchFactor(0, 1)

    def log_message(self, message: str) -> None:
        self.message_log.append(message)
        self.message_log.moveCursor(QTextCursor.MoveOperation.End)

    # ================= #
    # Control ROI Panel #
    # ================= #

    def setup_control_panel(self):
        self.control_panel = QHBoxLayout()
        self.layout.addLayout(self.control_panel)

        self.roi_button = QPushButton("Drag a ROI")
        self.roi_button.clicked.connect(self.start_drawing_roi)
        self.control_panel.addWidget(self.roi_button)

    def start_drawing_roi(self):
        self.video_view.start_drawing_roi()

    # ========== #
    # Plot View  #
    # ========== #

    def setup_plot_view(self):
        self.plot_view = PlotView()
        self.splitter.addWidget(self.plot_view)  # Add plot view to the splitter
        self.splitter.setStretchFactor(1, 1)  # Make plot view take less space initially

        # Add the progress bar layout to the splitter
        self.splitter.addWidget(QWidget())
        self.splitter.widget(2).setLayout(self.progress_bar_layout)

    def update_plot(self, value):
        self.plot_view.update_plot(value)

    def main(self):
        self.show()
        self.setFocus()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoLoaderApp()
    window.main()
    sys.exit(app.exec())
