"""
Simple module with a class to manage the data used in the analysis
"""

# pylint: disable=too-few-public-methods
class Logger:
    """
    Class to print in colour
    """
    DEBUG = '\033[96m'
    INFO = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    FATAL = '\33[101m'
    ENDC = '\033[0m'

    def __init__(self, text, level):
        """
        Initialize the class
        Parameters
        ------------------------------------------------
        text: str
            Text to be printed
        level: str
            Level of logger, possible values [DEBUG, INFO, WARNING, ERROR, FATAL, RESULT]
        """
        self._text_ = text
        self._level_ = level

        if level == 'DEBUG':
            print(f'{Logger.DEBUG}DEBUG{Logger.ENDC}: {text}')
        elif level == 'INFO':
            print(f'{Logger.INFO}INFO{Logger.ENDC}: {text}')
        elif level == 'WARNING':
            print(f'{Logger.WARNING}WARNING{Logger.ENDC}: {text}')
        elif level == 'ERROR':
            print(f'{Logger.ERROR}ERROR{Logger.ENDC}: {text}')
        elif level == 'FATAL':
            raise RuntimeError(text)
        elif level == 'RESULT':
            print(f'\n\n{text}\n\n')
        else:
            print(text)
