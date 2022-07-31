"""
Simple module with a class to manage the data used in the analysis
"""

class Colour_print:
    """
    Class to print in colour
    """    
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    
    def __init__(self, text, colour):
        """
        Initialize the class
        Parameters
        ------------------------------------------------
        text: str
            Text to be printed
        colour: str
            Colour of the text
        """
        self._text_ = text
        self._colour_ = colour
        
        if 'OK' in colour:
            print(f'{Colour_print.OKGREEN}{text}{Colour_print.ENDC}')
        elif colour == 'WARNING':
            print(f'{Colour_print.WARNING}{text}{Colour_print.ENDC}')
        elif colour == 'FAIL':
            print(f'{Colour_print.FAIL}{text}{Colour_print.ENDC}')
        else:
            print(f'{text}')