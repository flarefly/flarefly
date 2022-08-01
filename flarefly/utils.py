"""
Simple module with a class to manage the data used in the analysis
"""

import sys

class Logger:
    """
    Class to print in colour
    """    
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    
    def __init__(self, text, level):
        """
        Initialize the class
        Parameters
        ------------------------------------------------
        text: str
            Text to be printed
        level: str
            Level of logger, possible values [DEBUG, INFO, WARNING, ERROR, FATAL]
        """
        self._text_ = text
        self._level_ = level
        
        if level == 'DEBUG':
            print(f'{Colour_print.DEBUG}DEBUG{Colour_print.ENDC}: {text}')
        elif level == 'INFO':
            print(f'{Colour_print.INFO}INFO{Colour_print.ENDC}: {text}')
        elif level == 'WARNING':
            print(f'{Colour_print.WARNING}WARNING{Colour_print.ENDC}: {text}')
        elif level == 'ERROR':
            print(f'{Colour_print.ERROR}ERROR{Colour_print.ENDC}: {text}')
        elif level == 'FATAL':
            print(f'{Colour_print.FATAL}FATAL{Colour_print.ENDC}: {text}')
            sys.exit(0)
        else:
            print(text)