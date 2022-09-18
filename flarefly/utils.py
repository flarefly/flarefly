"""
Simple module with a class to manage the data used in the analysis
"""

import sys
from particle import Particle

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
            print(f'{Logger.FATAL}FATAL{Logger.ENDC}: {text}')
            sys.exit(0)
        elif level == 'RESULT':
            print(f'\n\n{text}\n\n')
        else:
            print(text)

def get_particle_mass(pdg_id=-999, pdg_name=''):
    """
    Get particle mass from PDG dictionary

    Parameters
    -------------------------------------------------
    pdg_name: str
        Name of particle
    pdg_id: int
        PDG id of particle

    Returns
    -------------------------------------------------
    mass: float
        Mass of particle
    """
    if pdg_id:
        return Particle.from_pdgid(pdg_id).mass
    if pdg_name:
        return Particle.from_name(pdg_name).mass

def get_particle_names(pdg_name):
    """
    Look for particle names containg pdg_name

    Parameters
    -------------------------------------------------
    pdg_name: str
        Name of particle to look for

    Returns
    -------------------------------------------------
    """

    print(f'All available particle containng "{pdg_name}" are:\n')
    for _, particle in enumerate(Particle.findall(pdg_name)):
        print(f'Name: {particle}, Mass: {particle.mass}, {particle.pdgid}')
