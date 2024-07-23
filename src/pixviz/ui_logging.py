import datetime
from typing import Literal

from PyQt6.QtGui import QTextCursor

__all__ = ['LOGGING_TYPE',
           'DEBUG_LOGGING',
           'log_message']

LOGGING_TYPE = Literal['DEBUG', 'INFO', 'IO', 'WARNING', 'ERROR']
DEBUG_LOGGING = False


def log_message(message: str, log_type: LOGGING_TYPE = 'INFO', debug_mode: bool = DEBUG_LOGGING) -> None:
    from main_gui import VideoLoaderApp

    app_instance = VideoLoaderApp().INSTANCE

    if not app_instance:
        print(message)
        return

    if not debug_mode and log_type == 'DEBUG':
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    color = _get_log_type_color(log_type)
    log_entry = f'<span style="color:{color};">[{timestamp}] [{log_type}] - {message}</span><br>'

    if app_instance.message_log is None:
        print(message)
    else:
        app_instance.message_log.insertHtml(log_entry)
        app_instance.message_log.moveCursor(QTextCursor.MoveOperation.End)


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
