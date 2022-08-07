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
        Get particle name
        """
        return self.pdg_dict()['name']

    def get_mass(self, unit='GeV'):
        """
        Get particle mass

        Parameters
        -------------------------------------------------
        unit: str
            Unit of the mass, possible values ['GeV', 'MeV']
        """
        if unit == 'GeV':
            return self.pdg_dict()['mass']
        if unit == 'MeV':
            return self.pdg_dict() * 1000
        Logger(f'Unit {unit} not supported', 'FATAL')

    def pdg_dict(self): # pylint: disable=too-many-statements, too-many-branches
        """
        Associate PDG code with particle name and mass

        Returns
        -------------------------------------------------
        pdg_dict: dict
            Dictionary with the particle name and mass
        """
        if self._pdg_ == 22:
            pdg_dict = {'name': 'photon', 'mass': 0.0}
        elif self._pdg_ == -2112:
            pdg_dict = {'name': 'anti-neutron', 'mass': 939.565}
        elif self._pdg_ == -11:
            pdg_dict = {'name': 'e-', 'mass': 0.510998928}
        elif self._pdg_ == -3122:
            pdg_dict = {'name': 'Lambda', 'mass': 1.115683}
        elif self._pdg_ == 11:
            pdg_dict = {'name': 'e+', 'mass': 0.510998928}
        elif self._pdg_ == -3222:
            pdg_dict = {'name': 'Sigma', 'mass': 1.18937}
        elif self._pdg_ == 12:
            pdg_dict = {'name': 'e-neutrino', 'mass': 0.0}
        elif self._pdg_ == -3212:
            pdg_dict = {'name': 'Sigma-', 'mass': 1.18937}
        elif self._pdg_ == -13:
            pdg_dict = {'name': 'mu+', 'mass': 0.1056583715}
        elif self._pdg_ == -3112:
            pdg_dict = {'name': 'Sigma-', 'mass': 1.18937}
        elif self._pdg_ == 13:
            pdg_dict = {'name': 'mu-', 'mass': 0.1056583715}
        elif self._pdg_ == -3322:
            pdg_dict = {'name': 'Xi0', 'mass': 1.32132}
        elif self._pdg_ == 111:
            pdg_dict = {'name': 'pi0', 'mass': 0.1349766}
        elif self._pdg_ == -3312:
            pdg_dict = {'name': 'Xi+', 'mass': 1.32132}
        elif self._pdg_ == 211:
            pdg_dict = {'name': 'pi+', 'mass': 0.13957018}
        elif self._pdg_ == -3334:
            pdg_dict = {'name': 'Omega+', 'mass': 1.67245}
        elif self._pdg_ == -211:
            pdg_dict = {'name': 'pi-', 'mass': 0.13957018}
        elif self._pdg_ == -15:
            pdg_dict = {'name': 'tau+', 'mass': 1.77682}
        elif self._pdg_ == 130:
            pdg_dict = {'name': 'KL0', 'mass': 0.497614}
        elif self._pdg_ == 15:
            pdg_dict = {'name': 'tau-', 'mass': 1.77682}
        elif self._pdg_ == 321:
            pdg_dict = {'name': 'K+', 'mass': 0.493677}
        elif self._pdg_ == 411:
            pdg_dict = {'name': 'D+', 'mass': 1.86962}
        elif self._pdg_ == -321:
            pdg_dict = {'name': 'K-', 'mass': 0.493677}
        elif self._pdg_ == -411:
            pdg_dict = {'name': 'D-', 'mass': 1.86962}
        elif self._pdg_ == 2112:
            pdg_dict = {'name': 'neutron', 'mass': 939.565}
        elif self._pdg_ == 421:
            pdg_dict = {'name': 'D0', 'mass': 1.8646}
        elif self._pdg_ == 2212:
            pdg_dict = {'name': 'proton', 'mass': 938.272}
        elif self._pdg_ == -421:
            pdg_dict = {'name': 'D0bar', 'mass': 1.8646}
        elif self._pdg_ == -2212:
            pdg_dict = {'name': 'anti-proton', 'mass': 938.272}
        elif self._pdg_ == 431:
            pdg_dict = {'name': 'D_s^+', 'mass': 1.96847}
        elif self._pdg_ == 310:
            pdg_dict = {'name': 'KS0', 'mass': 0.497614}
        elif self._pdg_ == -431:
            pdg_dict = {'name': 'D_s^-', 'mass': 1.96847}
        elif self._pdg_ == 4122:
            pdg_dict = {'name': 'Lambda_c+', 'mass': 2.2849}
        elif self._pdg_ == 3122:
            pdg_dict = {'name': 'Lambda0', 'mass': 1.115683}
        elif self._pdg_ == 24:
            pdg_dict = {'name': 'W+', 'mass': 80.399}
        elif self._pdg_ == 3222:
            pdg_dict = {'name': 'Sigma+', 'mass': 1.18937}
        elif self._pdg_ == -24:
            pdg_dict = {'name': 'W-', 'mass': 80.399}
        elif self._pdg_ == 3212:
            pdg_dict = {'name': 'Sigma0', 'mass': 1.18937}
        elif self._pdg_ == 23:
            pdg_dict = {'name': 'Z0', 'mass': 91.1876}
        elif self._pdg_ == 3112:
            pdg_dict = {'name': 'Sigma+', 'mass': 1.18937}
        elif self._pdg_ == 3322:
            pdg_dict = {'name': 'Xi0', 'mass': 1.32132}
        elif self._pdg_ == 3312:
            pdg_dict = {'name': 'Xi-', 'mass': 1.32132}
        elif self._pdg_ == 3334:
            pdg_dict = {'name': 'Omega-', 'mass': 1.67245}
        else:
            pdg_dict = {'name': 'unknown', 'mass': 0.0}
            Logger('Unknown particle', 'FATAL')
        return pdg_dict
