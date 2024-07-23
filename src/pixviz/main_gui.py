import json
import re
import sys
from pathlib import Path
from typing import ClassVar, Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QUrl, QRectF, QTimer, pyqtSlot
from PyQt6.QtGui import QPen, QColor, QKeyEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QWidget, QHBoxLayout, QSlider, QTextEdit, QGraphicsRectItem,
    QSplitter, QDialog, QProgressBar, QTableWidget, QTableWidgetItem
)
from matplotlib import pyplot as plt

from pixviz.ui_logging import log_message
from pixviz.roi import RoiLabelObject
from pixviz.ui_components import (
    FrameRateDialog,
    RoiSettingsDialog,
    VideoGraphicsView,
    PlotView,
    FrameProcessor
)

__all__ = ['run_gui']


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

        # reload
        self.reload_mode: bool = False

        # container for roi_name:elements in QGraphicsVideoItem
        self.rois: dict[str, RoiLabelObject] = {}

        #
        self.setup_layout()
        self.setup_controller()
        self._enable_button_load(False)

        self.timer = QTimer()
        self.timer.timeout.connect(self.video_view_process)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

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
        QPushButton:disabled {
            background-color: #555;
            color: #888;
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
        self.video_view.focusOutEvent = self._on_focus_out_event

    def _on_focus_out_event(self, event):
        """focus after drag roi"""
        self.setFocus()

    def _enable_button_load(self, enable: bool) -> None:
        """Enable or disable some buttons before/after load video"""
        self.play_button.setEnabled(enable)
        self.pause_button.setEnabled(enable)

        if not self.reload_mode:
            self.roi_button.setEnabled(enable)
            self.delete_roi_button.setEnabled(enable)
            self.process_button.setEnabled(enable)

    def load_video(self) -> None:
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

            log_message(f'Loaded Video: {file_path}', log_type='IO')

            # Prompt for the sampling rate
            dialog = FrameRateDialog(default_value=self.frame_rate)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.media_player.setSource(QUrl.fromLocalFile(file_path))
                self.frame_rate = dialog.get_sampling_rate()
                log_message(f'total frames: {self.total_frames}, frame_rate: {self.frame_rate}')
                self.media_player.pause()
                self.media_player.setPosition(0)
                self._enable_button_load(True)
            else:
                log_message('Video loading cancel')

    # ================= #
    # VideoGraphicsView #
    # ================= #

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
        """numpy array data output path"""
        file = Path(self.video_path)
        return file.with_stem(f'{file.stem}_pixviz').with_suffix('.npy')

    @property
    def meta_output_file(self) -> Path:
        """meta output path"""
        file = Path(self.video_path)
        return file.with_stem(f'{file.stem}_pixviz_meta').with_suffix('.json')

    def play_video(self) -> None:
        log_message("play", log_type='DEBUG')
        self.media_player.play()
        self.timer.start(int(1000 // self.frame_rate))

    def pause_video(self) -> None:
        log_message("pause", log_type='DEBUG')
        self.media_player.pause()
        self.timer.stop()

    def _handle_media_status(self, status) -> None:
        match status:
            case QMediaPlayer.MediaStatus.EndOfMedia:
                log_message("End of media reached", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.InvalidMedia:
                log_message("Invalid media", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.NoMedia:
                log_message("No media loaded", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.LoadingMedia:
                log_message("Loading media...", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.LoadedMedia:
                log_message("Media loaded", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.BufferedMedia:
                log_message("Media buffered", log_type='DEBUG')
            case QMediaPlayer.MediaStatus.StalledMedia:
                log_message("Media playback stalled", log_type='DEBUG')
            case _:
                log_message(f'unknown {status}', log_type='DEBUG')

    def update_duration(self, duration: int) -> None:
        self.progress_bar.setRange(0, duration)

    def update_position(self, position: int) -> None:
        self.progress_bar.setValue(position)
        self.update_frame_number(position)

    def update_frame_number(self, position: int) -> None:
        frame_number = int((position / 1000.0) * self.frame_rate)
        self.video_view.frame_label.setText(f"Frame: {frame_number}")

        if self.plot_view.enable_axvline:
            self.plot_view.update_vertical_line_position(frame_number)

    def set_position(self, position: int) -> None:
        self.media_player.setPosition(position)

    def video_view_process(self) -> None:
        """realtime proc each frame"""
        self.video_view.process_frame()

    # ===================== #
    # ROI Setting and Table #
    # ===================== #

    def start_drawing_roi(self) -> None:
        log_message('Enable Drag mode')
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
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # non-editable
            self.roi_table.setItem(row, 0, name_item)

            rect_item = QTableWidgetItem(str(roi.rect_item.rect()))
            rect_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # non-editable
            self.roi_table.setItem(row, 1, rect_item)

            func_item = QTableWidgetItem(roi.func)
            func_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # non-editable
            self.roi_table.setItem(row, 2, func_item)

    def delete_selected_roi(self) -> None:
        """delete the selected roi using the button click"""
        selected_items = self.roi_table.selectedItems()
        if not selected_items:
            log_message('No ROI selected for deletion.', log_type='ERROR')
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

        log_message(f"Deleted ROI: {roi_name}")

    # ================== #
    # Batch Process Mode #
    # ================== #

    def process_all_frames(self) -> None:
        """Process all the frames with the selected ROIs"""
        if len(self.rois) == 0:
            log_message('Please set an ROI first.', log_type='ERROR')
            return

        self.plot_view.realtime_proc = False
        self.plot_view.clear_all()

        for name in self.rois.keys():
            self.plot_view.add_axes(name)

        self.update_frame_number(0)
        self._enable_all_buttons(False)

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
            log_message('roi name index incorrect', log_type='ERROR')

        self._save_meta()

        n_rois = len(frame_values)
        ret = np.zeros((n_rois, self.total_frames))
        for i, dat in enumerate(frame_values.values()):
            ret[i] = dat

        np.save(self.data_output_file, ret)
        log_message(f'Pixel intensity value saved to directory: {self.data_output_file.parent}', log_type='IO')
        self._enable_all_buttons(True)

    def _save_meta(self):
        ret = {}
        for i, (name, roi) in enumerate(self.rois.items()):
            ret[name] = roi.to_meta(i)

        with self.meta_output_file.open('w') as f:
            json.dump(ret, f, sort_keys=True, indent=4)

    def _enable_all_buttons(self, enable: bool) -> None:
        """
        Enable or disable all the button.

        :param enable: bool
        """
        widgets = (
            self.load_video_button,
            self.load_result_button,
            self.roi_button,
            self.delete_roi_button,
            self.play_button,
            self.pause_button,
            self.process_button
        )

        for it in widgets:
            it.setEnabled(enable)

    # ============= #
    # Result Reload #
    # ============= #

    def load_result(self) -> None:
        """reload the result"""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Pixviz Result File (*.npy)')
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            log_message(f'Loaded Result: {file_path}')
            self.reload_from_file(file_path)

    def reload_from_file(self, file: str) -> None:
        """
        Reload data from the output file

        Require ``*meta.json`` in the same directory

        :param file: ``.npy dataset``
        :return:
        """
        file = Path(file)
        dat = np.load(file)
        meta_file = file.with_stem(f'{file.stem}_meta').with_suffix('.json')
        with open(meta_file, 'r') as file:
            meta = json.load(file)

        #
        self.reload_mode = True
        self.plot_view.enable_axvline = True
        self.plot_view.set_axvline()
        self._reload(meta, dat)

        for name, line in self.plot_view._roi_lines.items():
            line.set_xdata(self.plot_view.x_data[name])
            line.set_ydata(self.plot_view.y_data[name])

        self.plot_view.ax.relim()
        self.plot_view.ax.autoscale_view()

        self.plot_view.canvas.draw()

    def _reload(self, meta: dict[str, Any],
                dat: np.ndarray):
        # clear existing ROIs from the scene and plot
        self.plot_view.clear_all()
        self.rois.clear()
        self.video_view.roi_object.clear()

        self.roi_table.setRowCount(len(meta))
        for i, (name, it) in enumerate(meta.items()):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # non-editable
            self.roi_table.setItem(i, 0, name_item)

            rect_item = QTableWidgetItem(it['item'])
            rect_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # non-editable
            self.roi_table.setItem(i, 1, rect_item)

            func_item = QTableWidgetItem(it['func'])
            func_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # non-editable
            self.roi_table.setItem(i, 2, func_item)

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

        self._disable_button_reload()

    def _disable_button_reload(self) -> None:
        """disable some button in `reload mode`"""
        self.roi_button.setEnabled(False)
        self.delete_roi_button.setEnabled(False)
        self.process_button.setEnabled(False)

    # ============== #
    # Other Controls #
    # ============== #

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """keyboard event handle"""
        if self.frame_rate is None:
            return

        current_position = self.media_player.position()
        frame_duration = 1000.0 / self.frame_rate  # ms

        match event.key():

            case Qt.Key.Key_Right:
                log_message('Right arrow key pressed')  # TODO check receiver
                new_position = current_position + (10 * frame_duration)
                self.set_position(int(new_position))
            case Qt.Key.Key_Left:
                log_message('Left arrow key pressed')  # TODO check receiver
                new_position = current_position - (10 * frame_duration)
                self.set_position(int(new_position))
            case Qt.Key.Key_Space:
                if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                    self.pause_video()
                else:
                    self.play_video()

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
