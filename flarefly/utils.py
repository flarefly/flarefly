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

pdg_dict = {'22': {'name': 'photon', 'mass': 0.0},
            '-2112': {'name': 'anti-neutron', 'mass': 939.565},
            '-11': {'name': 'e-', 'mass': 0.510998928},
            '-3122': {'name': 'Lambda', 'mass': 1.115683},
            '11': {'name': 'e+', 'mass': 0.510998928},
            '-3222': {'name': 'Sigma', 'mass': 1.18937},
            '12': {'name': 'e-neutrino', 'mass': 0.0},
            '-3212': {'name': 'Sigma-', 'mass': 1.18937},
            '-13': {'name': 'mu+', 'mass': 0.1056583715},
            '-3112': {'name': 'Sigma-', 'mass': 1.18937},
            '13': {'name': 'mu-', 'mass': 0.1056583715},
            '-3322': {'name': 'Xi0', 'mass': 1.32132},
            '111': {'name': 'pi0', 'mass': 0.1349766},
            '-3312': {'name': 'Xi+', 'mass': 1.32132},
            '211': {'name': 'pi+', 'mass': 0.13957018},
            '-3334': {'name': 'Omega+', 'mass': 1.67245},
            '-211': {'name': 'pi-', 'mass': 0.13957018},
            '-15': {'name': 'tau+', 'mass': 1.77682},
            '130': {'name': 'KL0', 'mass': 0.497614},
            '15': {'name': 'tau-', 'mass': 1.77682},
            '321': {'name': 'K+', 'mass': 0.493677},
            '411': {'name': 'D+', 'mass': 1.86962},
            '-321': {'name': 'K-', 'mass': 0.493677},
            '-411': {'name': 'D-', 'mass': 1.86962},
            '2112': {'name': 'neutron', 'mass': 939.565},
            '421': {'name': 'D0', 'mass': 1.8646},
            '2212': {'name': 'proton', 'mass': 938.272},
            '-421': {'name': 'D0bar', 'mass': 1.8646},
            '-2212': {'name': 'anti-proton', 'mass': 938.272},
            '431': {'name': 'D_s+', 'mass': 1.96847},
            '310': {'name': 'KS0', 'mass': 0.497614},
            '-431': {'name': 'D_s^-', 'mass': 1.96847},
            '4122': {'name': 'Lambda_c+', 'mass': 2.2849},
            '3122': {'name': 'Lambda0', 'mass': 1.115683},
            '2': {'name': 'W+', 'mass': 80.399},
            '3222': {'name': 'Sigma+', 'mass': 1.18937},
            '-24': {'name': 'W-', 'mass': 80.399},
            '3212': {'name': 'Sigma0', 'mass': 1.18937},
            '23': {'name': 'Z0', 'mass': 91.1876},
            '3112': {'name': 'Sigma+', 'mass': 1.18937},
            '3322': {'name': 'Xi0', 'mass': 1.32132},
            '3312': {'name': 'Xi-', 'mass': 1.32132},
            '3334': {'name': 'Omega-', 'mass': 1.67245}}

def get_particle_mass(pdg_id=None, pdg_name=None): # pylint: disable=inconsistent-return-statements
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
        return pdg_dict[str(pdg_id)]['mass']
    if pdg_name:
        for key in pdg_dict:
            if pdg_dict[key]['name'] == pdg_name:
                return pdg_dict[key]['mass']
