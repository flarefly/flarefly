"""
Simple module with a class to manage the data used in the analysis
"""

import sys

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

class MCparticle:
    """
    Class containg MC particle informations
    """
    def __init__(self, pdg):
        """
        Initialize the class
        Parameters
        ------------------------------------------------
        pdg:int
            PDG code of the particle
        """
        self._pdg_ = pdg
        self._name_ = self.get_name()
        self._mass_ = self.get_mass()

    def get_name(self):
        """
        Get the name of the particle
        """
        return self.pdg_dict()[0]

    def get_mass(self, unit='GeV'):
        """
        Get the mass of the particle
        """
        if unit == 'GeV':
            return self.pdg_dict()[1]
        if unit == 'MeV':
            return self.pdg_dict()[1] * 1000
        Logger(f'Unit {unit} not supported', 'FATAL')

    def pdg_dict(self):
        """
        Create a dictionary with the PDG code and particle information
        """
        if self._pdg_ == 22:
            return  ('photon', 0.0)
        if self._pdg_ == -2112:
            return  ('anti-neutron', 939.565)
        if self._pdg_ == -11:
            return  ('e+', 0.510998928)
        if self._pdg_ == -3122:
            return  ('anti-Lambda', 1.115683)
        if self._pdg_ == 11:
            return  ('e-', 0.510998928)
        if self._pdg_ == -3222:
            return  ('Sigma-', 1.18937)
        if self._pdg_ == 12:
            return  ('e-neutrino', 0.0)
        if self._pdg_ == -3212:
            return  ('Sigma0', 1.18937)
        if self._pdg_ == -13:
            return  ('mu+', 105.6583745)
        if self._pdg_ == -3112:
            return  ('anti-Sigma-', 1.18937)
        if self._pdg_ == 13:
            return  ('mu-', 105.6583745)
        if self._pdg_ == -3322:
            return  ('Xi0', 1.32171)
        if self._pdg_ == 111:
            return  ('pi0', 0.135)
        if self._pdg_ == -3312:
            return  ('Sigma-', 1.18937)
        if self._pdg_ == 211:
            return  ('pi+', 0.13957018)
        if self._pdg_ == -3334:
            return  ('Omega-', 1.67245)
        if self._pdg_ == -211:
            return  ('pi-', 0.13957018)
        if self._pdg_ == -15:
            return  ('tau+', 1.77682)
        if self._pdg_ == 130:
            return  ('KL0', 0.497614)
        if self._pdg_ == 15:
            return  ('tau-', 1.77682)
        if self._pdg_ == 321:
            return  ('K+', 0.493677)
        if self._pdg_ == 411:
            return  ('D+', 1.86957)
        if self._pdg_ == -321:
            return  ('K-', 0.493677)
        if self._pdg_ == -411:
            return  ('D-', 1.86957)
        if self._pdg_ == 2112:
            return  ('neutron', 939.565)
        if self._pdg_ == 421:
            return  ('D0', 1.86484)
        if self._pdg_ == 2212:
            return  ('proton', 938.272013)
        if self._pdg_ == -421:
            return  ('anti-D0', 1.86484)
        if self._pdg_ == -2212:
            return  ('anti-proton', 938.272013)
        if self._pdg_ == 431:
            return  ('D_s+', 1.96847)
        if self._pdg_ == 310:
            return  ('KL_0', 0.497614)
        if self._pdg_ == -431:
            return  ('anti-D_s-', 1.96847)
        if self._pdg_ == 221:
            return  ('eta', 0.54785)
        if self._pdg_ == 4122:
            return  ('Lambda', 1.115683)
        if self._pdg_ == 3122:
            return  ('Lambda', 1.115683)
        if self._pdg_ == 24:
            return  ('W+', 80.385)
        if self._pdg_ == 3222:
            return  ('Sigma+', 1.18937)
        if self._pdg_ == -24:
            return  ('W-', 80.385)
        if self._pdg_ == 3212:
            return  ('Sigma0', 1.18937)
        if self._pdg_ == 23:
            return  ('Z', 91.1876)
        if self._pdg_ == 3112:
            return  ('Sigma+', 1.18937)
        if self._pdg_ == 3322:
            return  ('Xi+', 1.32171)
        if self._pdg_ == 3312:
            return  ('Sigma+', 1.18937)
        if self._pdg_ == 3334:
            return  ('Omega+', 1.67245)
        Logger('Unknown particle', 'ERROR')
        return ('unknown', 0.0)
