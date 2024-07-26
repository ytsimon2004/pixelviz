import datetime
from typing import Literal

from PyQt6.QtGui import QTextCursor

__all__ = ['LOGGING_TYPE',
           'DEBUG_LOGGING',
           'log_message']

LOGGING_TYPE = Literal['DEBUG', 'INFO', 'IO', 'WARNING', 'ERROR']
DEBUG_LOGGING = False


def log_message(message: str, log_type: LOGGING_TYPE = 'INFO',
                debug_mode: bool = DEBUG_LOGGING) -> None:
    """
    Logging in the message area of the GUI

    :param message: message string
    :param log_type: ``LOGGING_TYPE``
    :param debug_mode: If show the debug type
    """
    from .main_gui import PixVizGUI
    app = PixVizGUI.INSTANCE

    if not debug_mode and log_type == 'DEBUG':
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    color = _get_log_type_color(log_type)
    log_entry = f'<span style="color:{color};">[{timestamp}] [{log_type}] - {message}</span><br>'

    if app.message_log is None:
        print(message)
    else:
        app.message_log.insertHtml(log_entry)
        app.message_log.moveCursor(QTextCursor.MoveOperation.End)


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
