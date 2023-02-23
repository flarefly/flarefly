"""
Module containing the class used to perform mass fits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnchoredText

import zfit
import uproot
from hist import Hist
import mplhep
from particle import Particle
from flarefly.utils import Logger
import flarefly.custom_pdfs as cpdf


# pylint: disable=too-many-instance-attributes, too-many-lines, too-many-public-methods
class F2MassFitter:
    """
    Class used to perform mass fits with the zfit library
    https://github.com/zfit/zfit
    """

    def __init__(self, data_handler, name_signal_pdf, name_background_pdf, name="", chi2_loss=False):
        """
        Initialize the F2MassFitter class
        Parameters
        -------------------------------------------------
        data_handler: flarefly.DataHandler
            The data handler containing the data to fit

        name_signal_pdf: list
            The list of names for the signal pdfs. The possible options are:

            - 'gaussian'

            - 'doublegaus'

            - 'crystalball'

            - 'doublecb'

            - 'cauchy'

            - 'voigtian'

            - 'kde_exact' (requires to set the datasample and options)

            - 'kde_grid' (requires to set the datasample and options)

            - 'kde_fft' (requires to set the datasample and options)

            - 'kde_isj' (requires to set the datasample and options)

        name_background_pdf: list
            The list of names of the background pdfs. The possible options are:

            - 'nobkg'

            - 'expo'

            - 'powlaw'

            - 'expopow'

            - 'chebpolN' (N is the order of the polynomial)

            - 'kde_exact' (requires to set the datasample and options)

            - 'kde_grid' (requires to set the datasample and options)

            - 'kde_fft' (requires to set the datasample and options)

            - 'kde_isj' (requires to set the datasample and options)

        name: str
            Optional name for the fitter,
            needed in case of multiple fitters defined in the same script

        chi2_loss: bool
            chi2 minimization if True, nll minmization else
            Default value to False
        """

        self._data_handler_ = data_handler
        self._name_signal_pdf_ = name_signal_pdf
        self._name_background_pdf_ = name_background_pdf
        self._signal_pdf_ = [None for _ in enumerate(name_signal_pdf)]
        self._kde_signal_sample_ = [None for _ in enumerate(name_signal_pdf)]
        self._kde_signal_option_ = [None for _ in enumerate(name_signal_pdf)]
        if self._name_background_pdf_[0] == "nobkg":
            self._background_pdf_ = []
            self._kde_bkg_sample_ = []
            self._kde_bkg_option_ = []
        else:
            self._background_pdf_ = [None for _ in enumerate(name_background_pdf)]
            self._kde_bkg_sample_ = [None for _ in enumerate(name_background_pdf)]
            self._kde_bkg_option_ = [None for _ in enumerate(name_background_pdf)]
        self._total_pdf_ = None
        self._total_pdf_binned_ = None
        self._fit_result_ = None
        self._init_sgn_pars_ = [{} for _ in enumerate(name_signal_pdf)]
        self._init_bkg_pars_ = [{} for _ in enumerate(name_signal_pdf)]
        self._limits_sgn_pars_ = [{} for _ in enumerate(name_signal_pdf)]
        self._limits_bkg_pars_ = [{} for _ in enumerate(name_signal_pdf)]
        self._fix_sgn_pars_ = [{} for _ in enumerate(name_signal_pdf)]
        self._fix_bkg_pars_ = [{} for _ in enumerate(name_signal_pdf)]
        self._sgn_pars_ = [{} for _ in enumerate(name_signal_pdf)]
        self._bkg_pars_ = [{} for _ in enumerate(name_background_pdf)]
        if self._name_background_pdf_[0] == "nobkg":
            self._fracs_ = [None for _ in range(len(name_signal_pdf) - 1)]
        else:
            self._fracs_ = [None for _ in range(len(name_signal_pdf) + len(name_background_pdf) - 1)]
        self._rawyield_ = [0. for _ in enumerate(name_signal_pdf)]
        self._rawyield_err_ = [0. for _ in enumerate(name_signal_pdf)]
        self._minimizer_ = zfit.minimize.Minuit(verbosity=7)
        self._name_ = name
        self._ndf_ = None
        self._chi2_loss_ = chi2_loss
        self._base_sgn_cmap_ = plt.cm.get_cmap('viridis', len(self._signal_pdf_) * 4)
        self._sgn_cmap_ = ListedColormap(self._base_sgn_cmap_(np.linspace(0.4, 0.65, len(self._signal_pdf_))))

        zfit.settings.advanced_warnings.all = False
        zfit.settings.changed_warnings.all = False

    # pylint: disable=too-many-branches, too-many-statements
    def __build_signal_pdfs(self, obs):
        """
        Helper function to compose the signal pdfs
        """

        for ipdf, pdf_name in enumerate(self._name_signal_pdf_):
            if pdf_name == 'gaussian':
                self._init_sgn_pars_[ipdf].setdefault('mu', 1.865)
                self._init_sgn_pars_[ipdf].setdefault('sigma', 0.010)
                self._fix_sgn_pars_[ipdf].setdefault('mu', False)
                self._fix_sgn_pars_[ipdf].setdefault('sigma', False)
                self._limits_sgn_pars_[ipdf].setdefault('mu', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('sigma', [0., 1.e6])
                self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_mu_signal{ipdf}', self._init_sgn_pars_[ipdf]['mu'],
                    self._limits_sgn_pars_[ipdf]['mu'][0], self._limits_sgn_pars_[ipdf]['mu'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['mu'])
                self._sgn_pars_[ipdf][f'{self._name_}_sigma_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_sigma_signal{ipdf}', self._init_sgn_pars_[ipdf]['sigma'],
                    self._limits_sgn_pars_[ipdf]['sigma'][0], self._limits_sgn_pars_[ipdf]['sigma'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['sigma'])
                self._signal_pdf_[ipdf] = zfit.pdf.Gauss(
                    obs=obs,
                    mu=self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'],
                    sigma=self._sgn_pars_[ipdf][f'{self._name_}_sigma_signal{ipdf}']
                )
            elif pdf_name == 'doublegaus':
                self._init_sgn_pars_[ipdf].setdefault('mu', 1.865)
                self._init_sgn_pars_[ipdf].setdefault('sigma1', 0.010)
                self._init_sgn_pars_[ipdf].setdefault('sigma2', 0.100)
                self._init_sgn_pars_[ipdf].setdefault('frac1', 0.9)
                self._fix_sgn_pars_[ipdf].setdefault('mu', False)
                self._fix_sgn_pars_[ipdf].setdefault('sigma1', False)
                self._fix_sgn_pars_[ipdf].setdefault('sigma2', False)
                self._fix_sgn_pars_[ipdf].setdefault('frac1', False)
                self._limits_sgn_pars_[ipdf].setdefault('mu', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('sigma1', [0., 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('sigma2', [0., 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('frac1', [0., 1.])
                self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_mu_signal{ipdf}', self._init_sgn_pars_[ipdf]['mu'],
                    self._limits_sgn_pars_[ipdf]['mu'][0], self._limits_sgn_pars_[ipdf]['mu'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['mu'])
                self._sgn_pars_[ipdf][f'{self._name_}_sigma1_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_sigma1_signal{ipdf}', self._init_sgn_pars_[ipdf]['sigma1'],
                    self._limits_sgn_pars_[ipdf]['sigma1'][0], self._limits_sgn_pars_[ipdf]['sigma1'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['sigma1'])
                self._sgn_pars_[ipdf][f'{self._name_}_sigma2_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_sigma2_signal{ipdf}', self._init_sgn_pars_[ipdf]['sigma2'],
                    self._limits_sgn_pars_[ipdf]['sigma2'][0], self._limits_sgn_pars_[ipdf]['sigma2'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['sigma1'])
                self._sgn_pars_[ipdf][f'{self._name_}_frac1_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_frac1_signal{ipdf}', self._init_sgn_pars_[ipdf]['frac1'],
                    self._limits_sgn_pars_[ipdf]['frac1'][0], self._limits_sgn_pars_[ipdf]['frac1'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['frac1'])
                self._signal_pdf_[ipdf] = cpdf.DoubleGauss(
                    obs=obs,
                    mu=self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'],
                    sigma1=self._sgn_pars_[ipdf][f'{self._name_}_sigma1_signal{ipdf}'],
                    sigma2=self._sgn_pars_[ipdf][f'{self._name_}_sigma2_signal{ipdf}'],
                    frac1=self._sgn_pars_[ipdf][f'{self._name_}_frac1_signal{ipdf}'],
                )
            elif pdf_name == 'crystalball':
                self._init_sgn_pars_[ipdf].setdefault('mu', 1.865)
                self._init_sgn_pars_[ipdf].setdefault('sigma', 0.010)
                self._init_sgn_pars_[ipdf].setdefault('alpha', 0.5)
                self._init_sgn_pars_[ipdf].setdefault('n', 1.)
                self._fix_sgn_pars_[ipdf].setdefault('mu', False)
                self._fix_sgn_pars_[ipdf].setdefault('sigma', False)
                self._fix_sgn_pars_[ipdf].setdefault('alpha', False)
                self._fix_sgn_pars_[ipdf].setdefault('n', False)
                self._limits_sgn_pars_[ipdf].setdefault('mu', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('sigma', [0., 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('alpha', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('n', [0., 1.e6])
                self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_mu_signal{ipdf}', self._init_sgn_pars_[ipdf]['mu'],
                    self._limits_sgn_pars_[ipdf]['mu'][0], self._limits_sgn_pars_[ipdf]['mu'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['mu'])
                self._sgn_pars_[ipdf][f'{self._name_}_sigma_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_sigma_signal{ipdf}', self._init_sgn_pars_[ipdf]['sigma'],
                    self._limits_sgn_pars_[ipdf]['sigma'][0], self._limits_sgn_pars_[ipdf]['sigma'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['sigma'])
                self._sgn_pars_[ipdf][f'{self._name_}_alpha_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_alpha_signal{ipdf}', self._init_sgn_pars_[ipdf]['alpha'],
                    self._limits_sgn_pars_[ipdf]['alpha'][0], self._limits_sgn_pars_[ipdf]['alpha'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['alpha'])
                self._sgn_pars_[ipdf][f'{self._name_}_n_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_n_signal{ipdf}', self._init_sgn_pars_[ipdf]['n'],
                    self._limits_sgn_pars_[ipdf]['n'][0], self._limits_sgn_pars_[ipdf]['n'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['n'])
                self._signal_pdf_[ipdf] = zfit.pdf.CrystalBall(
                    obs=obs,
                    mu=self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'],
                    sigma=self._sgn_pars_[ipdf][f'{self._name_}_sigma_signal{ipdf}'],
                    alpha=self._sgn_pars_[ipdf][f'{self._name_}_alpha_signal{ipdf}'],
                    n=self._sgn_pars_[ipdf][f'{self._name_}_n_signal{ipdf}']
                )
            elif pdf_name == 'doublecb':
                self._init_sgn_pars_[ipdf].setdefault('mu', 1.865)
                self._init_sgn_pars_[ipdf].setdefault('sigma', 0.010)
                self._init_sgn_pars_[ipdf].setdefault('alphal', 0.5)
                self._init_sgn_pars_[ipdf].setdefault('nl', 1.)
                self._init_sgn_pars_[ipdf].setdefault('alphar', 0.5)
                self._init_sgn_pars_[ipdf].setdefault('nr', 1.)
                self._fix_sgn_pars_[ipdf].setdefault('mu', False)
                self._fix_sgn_pars_[ipdf].setdefault('sigma', False)
                self._fix_sgn_pars_[ipdf].setdefault('alphal', False)
                self._fix_sgn_pars_[ipdf].setdefault('nl', False)
                self._fix_sgn_pars_[ipdf].setdefault('alphar', False)
                self._fix_sgn_pars_[ipdf].setdefault('nr', False)
                self._limits_sgn_pars_[ipdf].setdefault('mu', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('sigma', [0., 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('alphal', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('nl', [0., 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('alphar', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('nr', [0., 1.e6])
                self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_mu_signal{ipdf}', self._init_sgn_pars_[ipdf]['mu'],
                    self._limits_sgn_pars_[ipdf]['mu'][0], self._limits_sgn_pars_[ipdf]['mu'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['mu'])
                self._sgn_pars_[ipdf][f'{self._name_}_sigma_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_sigma_signal{ipdf}', self._init_sgn_pars_[ipdf]['sigma'],
                    self._limits_sgn_pars_[ipdf]['sigma'][0], self._limits_sgn_pars_[ipdf]['sigma'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['sigma'])
                self._sgn_pars_[ipdf][f'{self._name_}_alphal_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_alphal_signal{ipdf}', self._init_sgn_pars_[ipdf]['alphal'],
                    self._limits_sgn_pars_[ipdf]['alphal'][0], self._limits_sgn_pars_[ipdf]['alphal'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['alphal'])
                self._sgn_pars_[ipdf][f'{self._name_}_nl_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_nl_signal{ipdf}', self._init_sgn_pars_[ipdf]['nl'],
                    self._limits_sgn_pars_[ipdf]['nl'][0], self._limits_sgn_pars_[ipdf]['nl'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['nl'])
                self._sgn_pars_[ipdf][f'{self._name_}_alphar_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_alphar_signal{ipdf}', self._init_sgn_pars_[ipdf]['alphar'],
                    self._limits_sgn_pars_[ipdf]['alphar'][0], self._limits_sgn_pars_[ipdf]['alphar'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['alphar'])
                self._sgn_pars_[ipdf][f'{self._name_}_nr_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_nr_signal{ipdf}', self._init_sgn_pars_[ipdf]['nr'],
                    self._limits_sgn_pars_[ipdf]['nr'][0], self._limits_sgn_pars_[ipdf]['nr'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['nr'])
                self._signal_pdf_[ipdf] = zfit.pdf.DoubleCB(
                    obs=obs,
                    mu=self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'],
                    sigma=self._sgn_pars_[ipdf][f'{self._name_}_sigma_signal{ipdf}'],
                    alphal=self._sgn_pars_[ipdf][f'{self._name_}_alphal_signal{ipdf}'],
                    nl=self._sgn_pars_[ipdf][f'{self._name_}_nl_signal{ipdf}'],
                    alphar=self._sgn_pars_[ipdf][f'{self._name_}_alphar_signal{ipdf}'],
                    nr=self._sgn_pars_[ipdf][f'{self._name_}_nr_signal{ipdf}']
                )
            elif pdf_name == 'cauchy':
                self._init_sgn_pars_[ipdf].setdefault('m', 1.865)
                self._init_sgn_pars_[ipdf].setdefault('gamma', 0.010)
                self._fix_sgn_pars_[ipdf].setdefault('m', False)
                self._fix_sgn_pars_[ipdf].setdefault('gamma', False)
                self._limits_sgn_pars_[ipdf].setdefault('m', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('gamma', [0., 1.e6])
                self._sgn_pars_[ipdf][f'{self._name_}_m_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_m_signal{ipdf}', self._init_sgn_pars_[ipdf]['m'],
                    self._limits_sgn_pars_[ipdf]['m'][0], self._limits_sgn_pars_[ipdf]['m'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['m'])
                self._sgn_pars_[ipdf][f'{self._name_}_gamma_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_gamma_signal{ipdf}', self._init_sgn_pars_[ipdf]['gamma'],
                    self._limits_sgn_pars_[ipdf]['gamma'][0], self._limits_sgn_pars_[ipdf]['gamma'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['gamma'])
                self._signal_pdf_[ipdf] = zfit.pdf.Cauchy(
                    obs=obs,
                    m=self._sgn_pars_[ipdf][f'{self._name_}_m_signal{ipdf}'],
                    gamma=self._sgn_pars_[ipdf][f'{self._name_}_gamma_signal{ipdf}']
                )
            elif pdf_name == 'voigtian':
                self._init_sgn_pars_[ipdf].setdefault('mu', 1.865)
                self._init_sgn_pars_[ipdf].setdefault('gamma', 0.010)
                self._init_sgn_pars_[ipdf].setdefault('sigma', 0.010)
                self._fix_sgn_pars_[ipdf].setdefault('mu', False)
                self._fix_sgn_pars_[ipdf].setdefault('gamma', False)
                self._fix_sgn_pars_[ipdf].setdefault('sigma', False)
                self._limits_sgn_pars_[ipdf].setdefault('mu', [0, 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('gamma', [0., 1.e6])
                self._limits_sgn_pars_[ipdf].setdefault('sigma', [0., 1.e6])
                self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_mu_signal{ipdf}', self._init_sgn_pars_[ipdf]['mu'],
                    self._limits_sgn_pars_[ipdf]['mu'][0], self._limits_sgn_pars_[ipdf]['mu'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['mu'])
                self._sgn_pars_[ipdf][f'{self._name_}_gamma_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_gamma_signal{ipdf}', self._init_sgn_pars_[ipdf]['gamma'],
                    self._limits_sgn_pars_[ipdf]['gamma'][0], self._limits_sgn_pars_[ipdf]['gamma'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['gamma'])
                self._sgn_pars_[ipdf][f'{self._name_}_sigma_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_sigma_signal{ipdf}', self._init_sgn_pars_[ipdf]['sigma'],
                    self._limits_sgn_pars_[ipdf]['sigma'][0], self._limits_sgn_pars_[ipdf]['sigma'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['sigma'])
                self._signal_pdf_[ipdf] = cpdf.Voigtian(
                    obs=obs,
                    mu=self._sgn_pars_[ipdf][f'{self._name_}_mu_signal{ipdf}'],
                    sigma=self._sgn_pars_[ipdf][f'{self._name_}_sigma_signal{ipdf}'],
                    gamma=self._sgn_pars_[ipdf][f'{self._name_}_gamma_signal{ipdf}']
                )
            elif 'kde' in pdf_name:
                if self._kde_signal_sample_[ipdf]:
                    if pdf_name == 'kde_exact':
                        self._signal_pdf_[ipdf] = zfit.pdf.KDE1DimExact(self._kde_signal_sample_[ipdf].get_data(),
                                                                        obs=self._kde_signal_sample_[ipdf].get_obs(),
                                                                        name=f'{self._name_}_kde_signal{ipdf}',
                                                                        **self._kde_signal_option_[ipdf])
                    elif pdf_name == 'kde_grid':
                        self._signal_pdf_[ipdf] = zfit.pdf.KDE1DimGrid(self._kde_signal_sample_[ipdf].get_data(),
                                                                       obs=self._kde_signal_sample_[ipdf].get_obs(),
                                                                       name=f'{self._name_}_kde_signal{ipdf}',
                                                                       **self._kde_signal_option_[ipdf])
                    elif pdf_name == 'kde_fft':
                        self._signal_pdf_[ipdf] = zfit.pdf.KDE1DimFFT(self._kde_signal_sample_[ipdf].get_data(),
                                                                      obs=self._kde_signal_sample_[ipdf].get_obs(),
                                                                      name=f'{self._name_}_kde_signal{ipdf}',
                                                                      **self._kde_signal_option_[ipdf])
                    elif pdf_name == 'kde_isj':
                        self._signal_pdf_[ipdf] = zfit.pdf.KDE1DimISJ(self._kde_signal_sample_[ipdf].get_data(),
                                                                      obs=self._kde_signal_sample_[ipdf].get_obs(),
                                                                      name=f'{self._name_}_kde_signal{ipdf}',
                                                                      **self._kde_signal_option_[ipdf])
                else:
                    Logger(f'Missing datasample for Kernel Density Estimation of signal {ipdf}!', 'FATAL')
            else:
                Logger(f'Signal pdf {pdf_name} not supported', 'FATAL')

    def __build_background_pdfs(self, obs):
        """
        Helper function to compose the background pdfs
        """

        for ipdf, pdf_name in enumerate(self._name_background_pdf_):
            if pdf_name == 'nobkg':
                Logger('Performing fit with no bkg pdf', 'WARNING')
                break
            if pdf_name == 'expo':
                self._init_bkg_pars_[ipdf].setdefault('lam', 0.1)
                self._limits_bkg_pars_[ipdf].setdefault('lam', [-1.e6, 1.e6])
                self._fix_bkg_pars_[ipdf].setdefault('lam', False)
                self._bkg_pars_[ipdf][f'{self._name_}_lam_bkg{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_lam_bkg{ipdf}', self._init_bkg_pars_[ipdf]['lam'],
                    self._limits_bkg_pars_[ipdf]['lam'][0], self._limits_bkg_pars_[ipdf]['lam'][1],
                    floating=not self._fix_bkg_pars_[ipdf]['lam'])
                self._background_pdf_[ipdf] = zfit.pdf.Exponential(
                    obs=obs,
                    lam=self._bkg_pars_[ipdf][f'{self._name_}_lam_bkg{ipdf}']
                )
            elif 'chebpol' in pdf_name:
                pol_degree = int(pdf_name.split('chebpol')[1])
                for deg in range(pol_degree + 1):
                    self._init_bkg_pars_[ipdf].setdefault(f'c{deg}', 0.1)
                    self._limits_bkg_pars_[ipdf].setdefault(f'c{deg}', [-1.e6, 1.e6])
                    self._fix_bkg_pars_[ipdf].setdefault(f'c{deg}', False)
                    self._bkg_pars_[ipdf][f'{self._name_}_c{deg}_bkg{ipdf}'] = zfit.Parameter(
                        f'{self._name_}_c{deg}_bkg{ipdf}', self._init_bkg_pars_[ipdf][f'c{deg}'],
                        self._limits_bkg_pars_[ipdf][f'c{deg}'][0], self._limits_bkg_pars_[ipdf][f'c{deg}'][1],
                        floating=not self._fix_bkg_pars_[ipdf][f'c{deg}'])
                coeff0 = self._bkg_pars_[ipdf][f'{self._name_}_c0_bkg{ipdf}']
                bkg_coeffs = [self._bkg_pars_[ipdf][f'{self._name_}_c{deg}_bkg{ipdf}']
                              for deg in range(1, pol_degree + 1)]
                self._background_pdf_[ipdf] = zfit.pdf.Chebyshev(obs=obs, coeffs=bkg_coeffs, coeff0=coeff0)
            elif 'powlaw' in pdf_name:
                self._init_bkg_pars_[ipdf].setdefault('mass', Particle.from_pdgid(211).mass*1e-3) # pion mass as default
                self._init_bkg_pars_[ipdf].setdefault('power', 1.)
                self._limits_bkg_pars_[ipdf].setdefault('mass', [0., 1.e6])
                self._limits_bkg_pars_[ipdf].setdefault('power', [-1.e6, 1.e6])
                self._fix_bkg_pars_[ipdf].setdefault('mass', True)
                self._fix_bkg_pars_[ipdf].setdefault('power', False)
                if self._data_handler_.get_limits()[0] < self._init_bkg_pars_[ipdf]["mass"]:
                    Logger('The mass parameter in powlaw cannot be smaller than the lower fit limit, please fix it.',
                           'FATAL')
                self._bkg_pars_[ipdf][f'{self._name_}_mass_bkg{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_mass_bkg{ipdf}', self._init_bkg_pars_[ipdf]['mass'],
                    self._limits_bkg_pars_[ipdf]['mass'][0], self._limits_bkg_pars_[ipdf]['mass'][1],
                    floating=not self._fix_bkg_pars_[ipdf]['mass'])
                self._bkg_pars_[ipdf][f'{self._name_}_power_bkg{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_power_bkg{ipdf}', self._init_bkg_pars_[ipdf]['power'],
                    self._limits_bkg_pars_[ipdf]['power'][0], self._limits_bkg_pars_[ipdf]['power'][1],
                    floating=not self._fix_bkg_pars_[ipdf]['power'])
                self._background_pdf_[ipdf] = cpdf.Pow(
                    obs=obs,
                    mass=self._bkg_pars_[ipdf][f'{self._name_}_mass_bkg{ipdf}'],
                    power=self._bkg_pars_[ipdf][f'{self._name_}_power_bkg{ipdf}']
                )
            elif 'expopow' in pdf_name:
                self._init_bkg_pars_[ipdf].setdefault('mass', Particle.from_pdgid(211).mass*1e-3) # pion mass as default
                self._init_bkg_pars_[ipdf].setdefault('lam', 0.1)
                self._limits_bkg_pars_[ipdf].setdefault('mass', [0., 1.e6])
                self._limits_bkg_pars_[ipdf].setdefault('lam', [-1.e6, 1.e6])
                self._fix_bkg_pars_[ipdf].setdefault('mass', True)
                self._fix_bkg_pars_[ipdf].setdefault('lam', False)
                if self._data_handler_.get_limits()[0] < self._init_bkg_pars_[ipdf]["mass"]:
                    Logger('The mass parameter in expopow cannot be smaller than the lower fit limit, please fix it.',
                           'FATAL')
                self._bkg_pars_[ipdf][f'{self._name_}_mass_bkg{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_mass_bkg{ipdf}', self._init_bkg_pars_[ipdf]['mass'],
                    self._limits_bkg_pars_[ipdf]['mass'][0], self._limits_bkg_pars_[ipdf]['mass'][1],
                    floating=not self._fix_bkg_pars_[ipdf]['mass'])
                self._bkg_pars_[ipdf][f'{self._name_}_lam_bkg{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_lam_bkg{ipdf}', self._init_bkg_pars_[ipdf]['lam'],
                    self._limits_bkg_pars_[ipdf]['lam'][0], self._limits_bkg_pars_[ipdf]['lam'][1],
                    floating=not self._fix_bkg_pars_[ipdf]['lam'])
                self._background_pdf_[ipdf] = cpdf.ExpoPow(
                    obs=obs,
                    mass=self._bkg_pars_[ipdf][f'{self._name_}_mass_bkg{ipdf}'],
                    lam=self._bkg_pars_[ipdf][f'{self._name_}_lam_bkg{ipdf}']
                )
            elif 'kde' in pdf_name:
                if self._kde_bkg_sample_[ipdf]:
                    if pdf_name == 'kde_exact':
                        self._background_pdf_[ipdf] = zfit.pdf.KDE1DimExact(self._kde_bkg_sample_[ipdf].get_data(),
                                                                            obs=self._kde_bkg_sample_[ipdf].get_obs(),
                                                                            name=f'{self._name_}_kde_bkg{ipdf}',
                                                                            **self._kde_bkg_option_[ipdf])
                    elif pdf_name == 'kde_grid':
                        self._background_pdf_[ipdf] = zfit.pdf.KDE1DimGrid(self._kde_bkg_sample_[ipdf].get_data(),
                                                                           obs=self._kde_bkg_sample_[ipdf].get_obs(),
                                                                           name=f'{self._name_}_kde_bkg{ipdf}',
                                                                           **self._kde_bkg_option_[ipdf])
                    elif pdf_name == 'kde_fft':
                        self._background_pdf_[ipdf] = zfit.pdf.KDE1DimFFT(self._kde_bkg_sample_[ipdf].get_data(),
                                                                          obs=self._kde_bkg_sample_[ipdf].get_obs(),
                                                                          name=f'{self._name_}_kde_bkg{ipdf}',
                                                                          **self._kde_bkg_option_[ipdf])
                    elif pdf_name == 'kde_isj':
                        self._background_pdf_[ipdf] = zfit.pdf.KDE1DimISJ(self._kde_bkg_sample_[ipdf].get_data(),
                                                                          obs=self._kde_bkg_sample_[ipdf].get_obs(),
                                                                          name=f'{self._name_}_kde_bkg{ipdf}',
                                                                          **self._kde_bkg_option_[ipdf])

                else:
                    Logger(f'Missing datasample for Kernel Density Estimation of background {ipdf}!', 'FATAL')
            else:
                Logger(f'Background pdf {pdf_name} not supported', 'FATAL')


    def __build_total_pdf(self):
        """
        Helper function to compose the total pdf
        """

        obs = self._data_handler_.get_obs()

        # order of the pdfs is signal, background

        self.__build_signal_pdfs(obs)
        self.__build_background_pdfs(obs)

        if len(self._signal_pdf_) + len(self._background_pdf_) == 1:
            self._total_pdf_ = self._signal_pdf_[0]
            return

        for ipdf, _ in enumerate(self._signal_pdf_):
            self._init_sgn_pars_[ipdf].setdefault('frac', 0.1)
            self._fix_sgn_pars_[ipdf].setdefault('frac', False)
            self._limits_sgn_pars_[ipdf].setdefault('frac', [0, 1.])
            if len(self._background_pdf_) == 0 and ipdf == len(self._signal_pdf_) - 1:
                continue
            self._fracs_[ipdf] = zfit.Parameter(f'{self._name_}_frac_signal{ipdf}',
                                                self._init_sgn_pars_[ipdf]['frac'],
                                                self._limits_sgn_pars_[ipdf]['frac'][0],
                                                self._limits_sgn_pars_[ipdf]['frac'][1],
                                                floating=not self._fix_sgn_pars_[ipdf]['frac'])

        if len(self._background_pdf_) > 1:
            for ipdf, _ in enumerate(self._background_pdf_):
                self._init_bkg_pars_[ipdf].setdefault('frac', 0.1)
                self._fix_bkg_pars_[ipdf].setdefault('frac', False)
                self._limits_bkg_pars_[ipdf].setdefault('frac', [0, 1.])
                self._fracs_[ipdf + len(self._signal_pdf_)] = zfit.Parameter(
                    f'{self._name_}_frac_bkg{ipdf}',
                    self._init_bkg_pars_[ipdf]['frac'],
                    self._limits_bkg_pars_[ipdf]['frac'][0],
                    self._limits_bkg_pars_[ipdf]['frac'][1],
                    floating=not self._fix_bkg_pars_[ipdf]['frac'])

        self._total_pdf_ = zfit.pdf.SumPDF(
            self._signal_pdf_+self._background_pdf_, self._fracs_)

    def __build_total_pdf_binned(self):
        """
        Helper function to compose the total pdf binned from unbinned
        """

        # for binned data, obs already contains the wanted binning
        if self._data_handler_.get_is_binned():
            obs = self._data_handler_.get_obs()
        # for unbinned data, one needs to impose a binning
        else:
            obs = self._data_handler_.get_binned_obs_from_unbinned_data()

        self._total_pdf_binned_ = zfit.pdf.BinnedFromUnbinnedPDF(self._total_pdf_, obs)

    def __prefit(self):
        """
        Helper function to perform a prefit to the sidebands
        """
        # pylint: disable=fixme
        #TODO: implement me
        Logger('Prefit step to be implemented', 'WARNING')

    def __get_all_fracs(self):
        """
        Helper function to get all fractions

        Returns
        -------------------------------------------------
        signal_fracs: list
            fractions of the signal pdfs
        background_fracs: list
            fractions of the background pdfs
        signal_err_fracs: list
            errors of fractions of the signal pdfs
        bkg_err_fracs: list
            errors of fractions of the background pdfs
        """
        signal_fracs, bkg_fracs, signal_err_fracs, bkg_err_fracs = ([] for _ in range(4))
        for frac_par in self._fracs_:
            par_name = frac_par.name
            if f'{self._name_}_frac_signal' in par_name:
                signal_fracs.append(self._fit_result_.params[par_name]['value'])
                signal_err_fracs.append(self._fit_result_.params[par_name]['hesse']['error'])
            elif f'{self._name_}_frac_bkg' in par_name:
                bkg_fracs.append(self._fit_result_.params[par_name]['value'])
                bkg_err_fracs.append(self._fit_result_.params[par_name]['hesse']['error'])

        return signal_fracs, bkg_fracs, signal_err_fracs, bkg_err_fracs

    def mass_zfit(self):
        """
        Perform a mass fit with the zfit library

        Returns
        -------------------------------------------------
        fit_result: zfit.minimizers.fitresult.FitResult
            The fit result
        """

        if self._data_handler_ is None:
            Logger('Data handler not specified', 'FATAL')

        self.__build_total_pdf()
        self.__build_total_pdf_binned()
        # pylint: disable=fixme
        self.__prefit() #TODO: implement me

        if self._data_handler_.get_is_binned():
            # chi2 loss
            if self._chi2_loss_:
                loss = zfit.loss.BinnedChi2(self._total_pdf_binned_,
                                            self._data_handler_.get_binned_data())
            # nll loss
            else:
                loss = zfit.loss.BinnedNLL(self._total_pdf_binned_,
                                            self._data_handler_.get_binned_data())
        else:
            loss = zfit.loss.UnbinnedNLL(
                model=self._total_pdf_, data=self._data_handler_.get_data())

        self._fit_result_ = self._minimizer_.minimize(loss=loss)
        if self._fit_result_.hesse() == {}:
            if self._fit_result_.hesse(method='hesse_np') == {}:
                Logger('Impossible to compute hesse error', 'FATAL')

        Logger(self._fit_result_, 'RESULT')

        norm = self._data_handler_.get_norm()
        if len(self._fracs_) == 0:
            self._rawyield_ = self._data_handler_.get_norm()
            self._rawyield_err_ = np.sqrt(self._rawyield_)
        else:
            for ipdf, _ in enumerate(self._signal_pdf_):
                if len(self._background_pdf_) > 0 or ipdf < len(self._signal_pdf_) - 1:
                    self._rawyield_[ipdf] = self._fit_result_.params[
                        f'{self._name_}_frac_signal{ipdf}']['value'] * norm
                    self._rawyield_err_[ipdf] = self._fit_result_.params[
                        f'{self._name_}_frac_signal{ipdf}']['hesse']['error'] * norm
                else:
                    frac, frac_err = 0., 0.
                    for ipdf2 in range(len(self._signal_pdf_)-1):
                        frac += self._fit_result_.params[
                            f'{self._name_}_frac_signal{ipdf2}']['value']
                        frac_err += np.sqrt(self._fit_result_.params[
                            f'{self._name_}_frac_signal{ipdf2}']['hesse']['error'])
                    self._rawyield_[ipdf] = frac * norm
                    self._rawyield_err_[ipdf] = frac_err * norm

        return self._fit_result_

    # pylint: disable=too-many-statements, too-many-locals
    def plot_mass_fit(self, **kwargs):
        """
        Plot the mass fit

        Parameters
        -------------------------------------------------
        **kwargs: dict
            Additional optional arguments:

            - style: str
                style to be used (see https://github.com/scikit-hep/mplhep for more details)

            - logy: bool
                log scale in y axis

            - figsize: tuple
                size of the figure

            - axis_title: str
                x-axis title

            - show_extra_info: bool
                show mu, sigma, chi2/ndf, signal, bkg, signal/bkg, significance

            - extra_info_massnsigma: float
                number of sigmas for extra info

            - extra_info_massnhwhm: float
                number of hwhms for extra info (alternative to extra_info_massnsigma)

            - extra_info_massrange: list
                mass range limits for extra info (alternative to extra_info_massnsigma)

            - extra_info_loc: list
                location of extra info (one for chi2 and one for other info)

            - num: int
                number of bins to plot pdfs converted into histograms

        Returns
        -------------------------------------------------
        fig: matplotlib.figure.Figure
            figure containing the mass fit plot
        """

        style = kwargs.get('style', 'LHCb2')
        logy = kwargs.get('logy', False)
        figsize = kwargs.get('figsize', (7, 7))
        bins = self._data_handler_.get_nbins()
        axis_title = kwargs.get('axis_title', self._data_handler_.get_var_name())
        show_extra_info = kwargs.get('show_extra_info', False)
        num = kwargs.get('num', 10000)
        mass_range = kwargs.get('extra_info_massrange', None)
        nhwhm = kwargs.get('extra_info_massnhwhm', None)
        nsigma = kwargs.get('extra_info_massnsigma', 3)
        loc = kwargs.get('extra_info_loc', ['upper left', 'lower right'])

        mplhep.style.use(style)

        obs = self._data_handler_.get_obs()
        limits = self._data_handler_.get_limits()

        fig, axs = plt.subplots(figsize=figsize)

        hdata = self._data_handler_.to_hist(lower_edge=limits[0],
                                            upper_edge=limits[1],
                                            nbins=bins,
                                            varname=self._data_handler_.get_var_name())

        hdata.plot(yerr=True, color='black', histtype='errorbar', label='data')
        bin_sigma = (limits[1] - limits[0]) / bins
        norm = self._data_handler_.get_norm() * bin_sigma

        x_plot = np.linspace(limits[0], limits[1], num=num)
        total_func = zfit.run(self._total_pdf_.pdf(x_plot, norm_range=obs))
        signal_funcs, signal_fracs, bkg_funcs, bkg_fracs = ([] for _ in range(4))
        for signal_pdf in self._signal_pdf_:
            signal_funcs.append(zfit.run(signal_pdf.pdf(x_plot, norm_range=obs)))
        for bkg_pdf in self._background_pdf_:
            bkg_funcs.append(zfit.run(bkg_pdf.pdf(x_plot, norm_range=obs)))
        for frac_par in self._fracs_:
            par_name = frac_par.name
            if f'{self._name_}_frac_signal' in par_name:
                signal_fracs.append(self._fit_result_.params[par_name]['value'])
            elif f'{self._name_}_frac_bkg' in par_name:
                bkg_fracs.append(self._fit_result_.params[par_name]['value'])
        if len(signal_fracs) == 0:
            signal_fracs.append(1.)

        # first draw backgrounds
        base_bkg_cmap = plt.cm.get_cmap('gist_heat', len(bkg_funcs) * 2)
        bkg_cmap = ListedColormap(base_bkg_cmap(np.linspace(0.3, 0.8, len(bkg_funcs))))
        for ibkg, bkg_func in enumerate(bkg_funcs):
            if ibkg < len(bkg_fracs) - 1:
                plt.plot(x_plot, bkg_func * norm * bkg_fracs[ibkg], color=bkg_cmap(ibkg),
                         ls='--', label=f'background {ibkg}')
            else:
                plt.plot(x_plot, bkg_func * norm * (1-sum(bkg_fracs)-sum(signal_fracs)),
                         color='firebrick', ls='--', label=f'background {ibkg}')
        # then draw signals
        for isgn, (frac, signal_func) in enumerate(zip(signal_funcs, signal_fracs)):
            plt.plot(x_plot, signal_func * norm * frac, color=self._sgn_cmap_(isgn))
            plt.fill_between(x_plot, signal_func * norm * frac, color=self._sgn_cmap_(isgn),
                             alpha=0.5, label=f'signal {isgn}')

        plt.plot(x_plot, total_func * norm, color='xkcd:blue', label='total fit')
        plt.xlim(limits[0], limits[1])
        plt.xlabel(axis_title)
        plt.ylabel(rf'counts / {(limits[1]-limits[0])/bins*1000:0.1f} MeV/$c^2$')
        plt.legend(loc='best')
        if logy:
            plt.yscale('log')
            plt.ylim(min(total_func) * norm / 5, max(total_func) * norm * 5)

        if show_extra_info:
            # info on chi2/ndf
            chi2 = self.get_chi2()
            ndf = self.get_ndf()
            anchored_text_chi2 = AnchoredText(fr'$\chi^2 / \mathrm{{ndf}} =${chi2:.2f} / {ndf}',
                                              loc = loc[0],
                                              frameon=False)
            # signal and background info for all signals
            text = []
            for idx, _ in enumerate(self._name_signal_pdf_):
                mass, mass_unc = self.get_mass(idx)
                sigma, sigma_unc = None, None
                gamma, gamma_unc = None, None
                if self._name_signal_pdf_[idx] in ['gaussian', 'crystalball', 'voigtian']:
                    sigma, sigma_unc = self.get_sigma(idx)
                if self._name_signal_pdf_[idx] in ['cauchy', 'voigtian']:
                    gamma, gamma_unc = self.get_signal_parameter(idx, 'gamma')
                extra_info = fr'signal{idx}''\n' + fr'  $\mu = {mass*1000:.1f}\pm{mass_unc*1000:.1f}$ MeV$/c^2$''\n'
                if sigma is not None:
                    extra_info += fr'  $\sigma = {sigma*1000:.1f}\pm{sigma_unc*1000:.1f}$ MeV$/c^2$''\n'
                if gamma is not None:
                    extra_info += fr'  $\Gamma = {gamma*1000:.1f}\pm{gamma_unc*1000:.1f}$ MeV$/c^2$''\n'
                if mass_range is not None:
                    signal, signal_err = self.get_signal(idx=idx, min=mass_range[0], max=mass_range[1])
                    bkg, bkg_err = self.get_background(idx=idx, min=mass_range[0], max=mass_range[1])
                    s_over_b, s_over_b_err = self.get_signal_over_background(idx=idx, min=mass_range[0],
                                                                             max=mass_range[1])
                    significance, significance_err = self.get_significance(idx=idx, min=mass_range[0],
                                                                           max=mass_range[1])
                    interval = f'[{mass_range[0]:.3f}, {mass_range[1]:.3f}]'
                    extra_info += fr'  $S({interval})={signal:.0f} \pm {signal_err:.0f}$''\n'
                    extra_info += fr'  $B({interval})={bkg:.0f} \pm {bkg_err:.0f}$''\n'
                    extra_info += fr'  $S/B({interval})={s_over_b:.2f} \pm {s_over_b_err:.2f}$''\n'
                    extra_info += fr'  Signif.$({interval})={significance:.1f} \pm {significance_err:.1f}$'
                elif nhwhm is not None:
                    signal, signal_err = self.get_signal(idx=idx, nhwhm=nhwhm)
                    bkg, bkg_err = self.get_background(idx=idx, nhwhm=nhwhm)
                    s_over_b, s_over_b_err = self.get_signal_over_background(idx=idx, nhwhm=nhwhm)
                    significance, significance_err = self.get_significance(idx=idx, nhwhm=nhwhm)
                    extra_info += fr'  $S=${signal:.0f} $\pm$ {signal_err:.0f}''\n'
                    extra_info += fr'  $B({nhwhm}~\mathrm{{HWHM}})=${bkg:.0f} $\pm$ {bkg_err:.0f}''\n'
                    extra_info += fr'  $S/B({nhwhm}~\mathrm{{HWHM}})=${s_over_b:.2f} $\pm$ {s_over_b_err:.2f}''\n'
                    extra_info += fr'  Signif.$({nhwhm}~\mathrm{{HWHM}})=${significance:.1f} $\pm$ {significance_err:.1f}'
                else:
                    signal, signal_err = self.get_signal(idx=idx, nsigma=nsigma)
                    bkg, bkg_err = self.get_background(idx=idx, nsigma=nsigma)
                    s_over_b, s_over_b_err = self.get_signal_over_background(idx=idx, nsigma=nsigma)
                    significance, significance_err = self.get_significance(idx=idx, nsigma=nsigma)
                    extra_info += fr'  $S=${signal:.0f} $\pm$ {signal_err:.0f}''\n'
                    extra_info += fr'  $B({nsigma}\sigma)=${bkg:.0f} $\pm$ {bkg_err:.0f}''\n'
                    extra_info += fr'  $S/B({nsigma}\sigma)=${s_over_b:.2f} $\pm$ {s_over_b_err:.2f}''\n'
                    extra_info += fr'  Signif.$({nsigma}\sigma)=${significance:.1f} $\pm$ {significance_err:.1f}'
            text.append(extra_info)
            concatenated_text = '\n'.join(text)
            anchored_text_signal = AnchoredText(concatenated_text, loc = loc[1], frameon=False)

            axs.add_artist(anchored_text_chi2)
            axs.add_artist(anchored_text_signal)

        return fig

    # pylint: disable=too-many-statements, too-many-locals
    def dump_to_root(self, filename, **kwargs):
        """
        Plot the mass fit

        Parameters
        -------------------------------------------------
        filename: str
            Name of output ROOT file

        **kwargs: dict
            Additional optional arguments:

            - axis_title: str
                x-axis title

            - num: int
                number of bins to plot pdfs converted into histograms

            - option: str
                option (recreate or update)


            - suffix: str
                suffix to append to objects
        """

        num = kwargs.get('num', 10000)
        suffix = kwargs.get('suffix', '')
        option = kwargs.get('option', 'recreate')
        bins = self._data_handler_.get_nbins()
        obs = self._data_handler_.get_obs()
        limits = self._data_handler_.get_limits()

        hdata = self._data_handler_.to_hist(lower_edge=limits[0],
                                            upper_edge=limits[1],
                                            nbins=bins,
                                            varname=self._data_handler_.get_var_name())
        # write data
        self.__write_data(hdata, f'hdata{suffix}', filename, option)

        bin_sigma = (limits[1] - limits[0]) / bins
        norm = self._data_handler_.get_norm() * bin_sigma
        x_plot = np.linspace(limits[0], limits[1], num=num)

        total_func = zfit.run(self._total_pdf_.pdf(x_plot, norm_range=obs))
        # write total_func
        self.__write_pdf(histname=f'total_func{suffix}', weight=total_func, num=num,
                         filename=filename, option='update')

        signal_funcs, signal_fracs, bkg_funcs, bkg_fracs = ([] for _ in range(4))
        for signal_pdf in self._signal_pdf_:
            signal_funcs.append(zfit.run(signal_pdf.pdf(x_plot, norm_range=obs)))
        for bkg_pdf in self._background_pdf_:
            bkg_funcs.append(zfit.run(bkg_pdf.pdf(x_plot, norm_range=obs)))
        for frac_par in self._fracs_:
            par_name = frac_par.name
            if f'{self._name_}_frac_signal' in par_name:
                signal_fracs.append(self._fit_result_.params[par_name]['value'])
            elif f'{self._name_}_frac_bkg' in par_name:
                bkg_fracs.append(self._fit_result_.params[par_name]['value'])
        if len(signal_fracs) == 0:
            signal_fracs.append(1.)

        # first write backgrounds
        for ibkg, bkg_func in enumerate(bkg_funcs):
            if ibkg < len(bkg_fracs) - 1:
                self.__write_pdf(histname=f'bkg_{ibkg}{suffix}',
                                 weight=bkg_func * norm * bkg_fracs[ibkg],
                                 num=num, filename=filename, option='update')
            else:
                self.__write_pdf(histname=f'bkg_{ibkg}{suffix}',
                               weight=bkg_func * norm * (1-sum(bkg_fracs)-sum(signal_fracs)),
                               num=num, filename=filename, option='update')
        # then write signals
        for isgn, (frac, signal_func) in enumerate(zip(signal_funcs, signal_fracs)):
            self.__write_pdf(histname=f'signal_{isgn}{suffix}',
                           weight=signal_func * norm * frac,
                           num=num, filename=filename, option='update')

    @property
    def get_fit_result(self):
        """
        Get the fit result

        Returns
        -------------------------------------------------
        fit_result: zfit.minimizers.fitresult.FitResult
            The fit result
        """
        return self._fit_result_

    def get_ndf(self):
        """
        Get the number of degrees of freedom for chi2 fit
        ndf = nbins - nfreeparams - 1
        -1 because the data sample size is fixed

        Returns
        -------------------------------------------------
        ndf: int
            The number of degrees of freedom
        """
        nbins = self._data_handler_.get_nbins()
        nfreeparams =  len(self._fit_result_.params)
        self._ndf_ = nbins - nfreeparams - 1
        return self._ndf_

    def get_chi2(self):
        """
        Get chi2 for binned data

        Returns
        -------------------------------------------------
        chi2: float
            chi2
        """

        chi2 = 0
        norm = self._data_handler_.get_norm()
        if self._data_handler_.get_is_binned():
            # for chi2 loss, just retrieve loss value in fit result
            if self._chi2_loss_:
                return float(self._fit_result_.loss.value())

            # for nll loss, compute chi2 "by hand"
            # access normalized data values and errors for all bins
            binned_data = self._data_handler_.get_binned_data()
            data_values = binned_data.values()
            data_variances = binned_data.variances()
            # access model predicted values
            model_values = self._total_pdf_binned_.values()*norm
            # compute chi2
            for (data, model, data_variance) in zip(data_values, model_values, data_variances):
                chi2 += (data - model)**2/data_variance
            return chi2

        # for unbinned data
        data_values = self._data_handler_.get_binned_data_from_unbinned_data()
        # access model predicted values
        model_values = self._total_pdf_binned_.values()*norm
        # compute chi2
        for (data, model) in zip(data_values, model_values):
            chi2 += (data - model)**2/data

        return float(chi2)

    def get_chi2_ndf(self):
        """
        Get the reduced chi2 (chi2 divided by number of degrees of freedom)
        for binned data

        Returns
        -------------------------------------------------
        chi2_ndf: float
            The reduced chi2
        """
        return self.get_chi2()/self.get_ndf()

    def get_raw_residuals(self):
        """
        Get the raw residuals (data_value - bkg_model_value) for all bins

        Returns
        -------------------------------------------------
        residuals: array[float]
            The residuals
        """

        bins = self._data_handler_.get_nbins()
        norm = self._data_handler_.get_norm()
        residuals = [None]*bins
        background_pdf_binned_ = [None for _ in enumerate(self._name_background_pdf_)]
        model_bkg_values = [None for _ in enumerate(self._name_background_pdf_)]

        # access normalized data values and errors for all bins
        if self._data_handler_.get_is_binned():
            binned_data = self._data_handler_.get_binned_data()
            data_values = binned_data.values()
            variances = binned_data.variances()
            obs = self._data_handler_.get_obs()
        else:
            data_values = self._data_handler_.get_binned_data_from_unbinned_data()
            variances = data_values # poissonian errors
            obs = self._data_handler_.get_binned_obs_from_unbinned_data()

        # get background fractions
        if len(self._background_pdf_) == 1:
            signal_fracs, _, _, _ = self.__get_all_fracs()
            bkg_fracs = [1 - sum(signal_fracs)]
        else:
            _, bkg_fracs, _, _ = self.__get_all_fracs()
        # access model predicted values for background
        for ipdf, _ in enumerate(self._name_background_pdf_):
            background_pdf_binned_[ipdf] = zfit.pdf.BinnedFromUnbinnedPDF(self._background_pdf_[ipdf], obs)
            model_bkg_values[ipdf] = background_pdf_binned_[ipdf].values()*bkg_fracs[ipdf]*norm
        # compute residuals
        for ibin, data in enumerate(data_values):
            residuals[ibin] = float(data)
            for ipdf, _ in enumerate(self._name_background_pdf_):
                residuals[ibin] -= model_bkg_values[ipdf][ibin]

        return residuals, variances

    def plot_raw_residuals(self, **kwargs):
        """
        Plot the raw residuals

        Parameters
        -------------------------------------------------
        **kwargs: dict
            Additional optional arguments:

            - style: str
                style to be used (see https://github.com/scikit-hep/mplhep for more details)

            - figsize: tuple
                size of the figure

            - axis_title: str
                x-axis title

        Returns
        -------------------------------------------------
        fig: matplotlib.figure.Figure
            figure containing the raw residuals plot
        """

        style = kwargs.get('style', 'LHCb2')
        figsize = kwargs.get('figsize', (7, 7))
        axis_title = kwargs.get('axis_title', self._data_handler_.get_var_name())

        mplhep.style.use(style)

        obs = self._data_handler_.get_obs()
        limits = self._data_handler_.get_limits()

        fig = plt.figure(figsize=figsize)

        residuals, variances = self.get_raw_residuals()
        # draw residuals
        plt.errorbar(
            self._data_handler_.get_bin_center(),
            residuals,
            xerr = None,
            yerr = np.sqrt(variances),
            linestyle = "None",
            elinewidth = 1,
            capsize = 0,
            color = "black",
            marker = "o",
            markersize = 5,
            label = "residuals"
        )
        bins = self._data_handler_.get_nbins()
        bin_sigma = (limits[1] - limits[0]) / bins
        norm = self._data_handler_.get_norm() * bin_sigma

        x_plot = np.linspace(limits[0], limits[1], num=1000)
        signal_funcs, signal_fracs, _, _ = ([] for _ in range(4))
        for signal_pdf in self._signal_pdf_:
            signal_funcs.append(zfit.run(signal_pdf.pdf(x_plot, norm_range=obs)))
        for frac_par in self._fracs_:
            par_name = frac_par.name
            if f'{self._name_}_frac_signal' in par_name:
                signal_fracs.append(self._fit_result_.params[par_name]['value'])
        if len(signal_fracs) == 0:
            signal_fracs.append(1.)

        # draw signals
        for isgn, (signal_func, frac) in enumerate(zip(signal_funcs, signal_fracs)):
            plt.plot(x_plot, signal_func * norm * frac, color=self._sgn_cmap_(isgn))
            plt.fill_between(x_plot, signal_func * norm * frac, color=self._sgn_cmap_(isgn),
                             alpha=0.5, label=f'signal {isgn}')

        plt.xlim(limits[0], limits[1])
        plt.xlabel(axis_title)
        plt.ylabel(rf'(data - fitted bkg) / {(limits[1]-limits[0])/bins*1000:0.1f} MeV/$c^2$')
        plt.legend(loc='best')

        return fig

    def get_std_residuals(self):
        """
        Get the standardized residuals
        (data_value - bkg_model_value)/ sigma_data for all bins

        Returns
        -------------------------------------------------
        residuals: array[float]
            The standardized residuals
        """

        bins = self._data_handler_.get_nbins()
        norm = self._data_handler_.get_norm()
        residuals, residuals_variances = [None]*bins, [None]*bins

        # access normalized data values and errors for all bins
        if self._data_handler_.get_is_binned():
            binned_data = self._data_handler_.get_binned_data()
            data_values = binned_data.values()
            variances = binned_data.variances()
        else:
            data_values = self._data_handler_.get_binned_data_from_unbinned_data()
            variances = data_values # poissonian errors

        # access model predicted values for background
        self.__build_total_pdf_binned()
        model_values = self._total_pdf_binned_.values()*norm
        for ibin, (data, model, variance) in enumerate(zip(data_values, model_values, variances)):
            if variance == 0:
                Logger('Null variance. Consider enlarging the bins.', 'FATAL')
            residuals[ibin] = float((data - model)/np.sqrt(variance))
            residuals_variances[ibin] = float(variance/np.sqrt(variance))

        return residuals, residuals_variances

    def plot_std_residuals(self, **kwargs):
        """
        Plot the raw residuals

        Parameters
        -------------------------------------------------
        **kwargs: dict
            Additional optional arguments:

            - style: str
                style to be used (see https://github.com/scikit-hep/mplhep for more details)

            - figsize: tuple
                size of the figure

            - axis_title: str
                x-axis title

        Returns
        -------------------------------------------------
        fig: matplotlib.figure.Figure
            figure containing the raw residuals plot
        """

        style = kwargs.get('style', 'LHCb2')
        figsize = kwargs.get('figsize', (7, 7))
        axis_title = kwargs.get('axis_title', self._data_handler_.get_var_name())

        mplhep.style.use(style)

        limits = self._data_handler_.get_limits()
        bins = self._data_handler_.get_nbins()
        bin_center = self._data_handler_.get_bin_center()

        fig = plt.figure(figsize=figsize)

        residuals, variances = self.get_std_residuals()
        # draw residuals
        plt.errorbar(bin_center,
                    residuals,
                    xerr = None,
                    yerr = np.sqrt(variances),
                    linestyle = "None",
                    elinewidth = 1,
                    capsize = 0,
                    color = "black",
                    marker = "o",
                    markersize = 5,
                    label = None)

        # line at 0
        plt.plot([bin_center[0], bin_center[-1]], [0., 0.], lw=2, color='xkcd:blue')
        # line at -3 sigma
        plt.plot([bin_center[0], bin_center[-1]], [-3., -3.], lw=2, color='xkcd:red')
        # line at 3 sigma
        plt.plot([bin_center[0], bin_center[-1]], [3., 3.], lw=2, color='xkcd:red')

        plt.xlim(limits[0], limits[1])
        plt.xlabel(axis_title)
        plt.ylabel(fr"$\dfrac{{ \mathrm{{data}} - \mathrm{{total \ fit}} }}{{ \sigma_{{ \mathrm{{data}} }} }}$"
                   fr"/ {(limits[1]-limits[0])/bins*1000:0.1f} MeV/$c^2$")

        return fig

    def get_raw_yield(self, idx=0):
        """
        Get the raw yield and its error

        Parameters
        -------------------------------------------------
        idx: int
            Index of the raw yield to be returned (default: 0)

        Returns
        -------------------------------------------------
        raw_yield: float
            The raw yield obtained from the fit
        raw_yield_err: float
            The raw yield error obtained from the fit
        """
        return self._rawyield_[idx], self._rawyield_err_[idx]

    def get_mass(self, idx=0):
        """
        Get the mass and its error

        Parameters
        -------------------------------------------------
        idx: int
            Index of the mass to be returned (default: 0)

        Returns
        -------------------------------------------------
        mass: float
            The mass value obtained from the fit
        mass_err: float
            The mass error obtained from the fit
        """
        mass_name = 'm' if self._name_signal_pdf_[idx] == 'cauchy' else 'mu'
        if self._fix_sgn_pars_[idx][mass_name]:
            mass = self._init_sgn_pars_[idx][mass_name]
            mass_err = 0.
        else:
            mass = self._fit_result_.params[f'{self._name_}_{mass_name}_signal{idx}']['value']
            mass_err = self._fit_result_.params[f'{self._name_}_{mass_name}_signal{idx}']['hesse']['error']

        return mass, mass_err

    def get_sigma(self, idx=0):
        """
        Get the sigma and its error

        Parameters
        -------------------------------------------------
        idx: int
            Index of the sigma to be returned (default: 0)

        Returns
        -------------------------------------------------
        sigma: float
            The sigma value obtained from the fit
        sigma_err: float
            The sigma error obtained from the fit
        """
        if self._name_signal_pdf_[idx] not in ['gaussian', 'crystalball', 'voigtian']:
            Logger(f'Sigma parameter not defined for {self._name_signal_pdf_[idx]} pdf!', 'ERROR')
            return 0., 0.

        if self._fix_sgn_pars_[idx]['sigma']:
            sigma = self._init_sgn_pars_[idx]['sigma']
            sigma_err = 0.
        else:
            sigma = self._fit_result_.params[f'{self._name_}_sigma_signal{idx}']['value']
            sigma_err = self._fit_result_.params[f'{self._name_}_sigma_signal{idx}']['hesse']['error']

        return sigma, sigma_err

    def get_hwhm(self, idx=0):
        """
        Get the half width half maximum and its error

        Parameters
        -------------------------------------------------
        idx: int
            Index of the sigma to be returned (default: 0)

        Returns
        -------------------------------------------------
        hwhm: float
            The sigma value obtained from the fit
        hwhm_err: float
            The sigma error obtained from the fit
        """
        if self._name_signal_pdf_[idx] not in ['gaussian', 'cauchy', 'voigtian']:
            Logger(f'HFWM parameter not defined for {self._name_signal_pdf_[idx]} pdf!', 'ERROR')
            return 0., 0.

        if self._name_signal_pdf_[idx] == 'gaussian':
            mult_fact = np.sqrt(2 * np.log(2))
            hwhm, hwhm_err = self.get_sigma(idx)
            hwhm *= mult_fact
            hwhm_err *= mult_fact
        elif self._name_signal_pdf_[idx] == 'cauchy':
            hwhm, hwhm_err = self.get_signal_parameter(idx, 'gamma')
        elif self._name_signal_pdf_[idx] == 'voigtian':
            mult_fact = np.sqrt(2 * np.log(2))
            sigma, sigma_err = self.get_sigma(idx)
            sigma *= mult_fact
            sigma_err *= mult_fact
            gamma, gamma_err = self.get_signal_parameter(idx, 'gamma')
            hwhm = 0.5346 * gamma + np.sqrt(0.2166 * gamma**2 + sigma**2)
            # we neglect the correlation between sigma and gamma
            der_sigma = sigma / np.sqrt(0.0721663 + sigma**2)
            der_gamma = 0.5346 + (0.2166 * gamma) / np.sqrt(0.2166 * gamma**2 + sigma**2)
            hwhm_err = np.sqrt(der_sigma**2 * sigma_err**2 + der_gamma**2 * gamma_err**2)

        return hwhm, hwhm_err

    def get_signal_parameter(self, idx, par):
        """
        Get a signal parameter and its error

        Parameters
        -------------------------------------------------
        idx: int
            Index of the parameter to be returned (default: 0)
        par: str
            parameter to return

        Returns
        -------------------------------------------------
        parameter: float
            The parameter value obtained from the fit
        parameter_err: float
            The parameter error obtained from the fit

        """

        if self._fix_sgn_pars_[idx][par]:
            parameter = self._init_sgn_pars_[idx][par]
            parameter_err = 0.
        else:
            parameter = self._fit_result_.params[f'{self._name_}_{par}_signal{idx}']['value']
            parameter_err = self._fit_result_.params[f'{self._name_}_{par}_signal{idx}']['hesse']['error']

        return parameter, parameter_err

    def get_background_parameter(self, idx, par):
        """
        Get a background parameter and its error

        Parameters
        -------------------------------------------------
        idx: int
            Index of the parameter to be returned (default: 0)
        par: str
            parameter to return

        Returns
        -------------------------------------------------
        parameter: float
            The parameter value obtained from the fit
        parameter_err: float
            The parameter error obtained from the fit

        """

        if self._fix_bkg_pars_[idx][par]:
            parameter = self._init_bkg_pars_[idx][par]
            parameter_err = 0.
        else:
            parameter = self._fit_result_.params[f'{self._name_}_{par}_bkg{idx}']['value']
            parameter_err = self._fit_result_.params[f'{self._name_}_{par}_bkg{idx}']['hesse']['error']

        return parameter, parameter_err

    def get_signal(self, idx=0, **kwargs):
        """
        Get the signal and its error in a given invariant-mass region

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal to be returned
        **kwargs: dict
            Additional optional arguments:

            - nsigma: float
                nsigma invariant-mass window around mean for signal computation

            - nhwhm: float
                number of hwhm invariant-mass window around mean for signal and background computation 
                (alternative to nsigma)

            - min: float
                minimum value of invariant-mass for signal computation (alternative to nsigma)

            - max: float
                maximum value of invariant-mass for signal computation (alternative to nsigma)

        Returns
        -------------------------------------------------
        signal: float
            The signal value obtained from the fit
        signal_err: float
            The signal error obtained from the fit
        """

        nsigma = kwargs.get('nsigma', 3.)
        nhwhm = kwargs.get('nhwhm', None)
        min_value = kwargs.get('min', None)
        max_value = kwargs.get('max', None)
        use_nsigma = True

        if nhwhm is not None and (min_value is not None or max_value is not None):
            Logger('I cannot compute the signal within a fixed mass interval and a number of HWFM', 'ERROR')
            return 0., 0.

        if min_value is not None and max_value is not None:
            use_nsigma = False

        if nhwhm is not None:
            use_nsigma = False
            if self._name_signal_pdf_[idx] not in ['gaussian', 'cauchy', 'voigtian']:
                Logger('HWHM not defined, I cannot compute the signal for this pdf', 'ERROR')
                return 0., 0.
            mass, _ = self.get_mass(idx)
            hwhm, _ = self.get_hwhm(idx)
            min_value = mass - nhwhm * hwhm
            max_value = mass + nhwhm * hwhm

        if use_nsigma:
            if self._name_signal_pdf_[idx] not in ['gaussian', 'crystalball']:
                Logger('Sigma not defined, I cannot compute the signal for this pdf', 'ERROR')
                return 0., 0.
            mass, _ = self.get_mass(idx)
            sigma, _ = self.get_sigma(idx)
            min_value = mass - nsigma * sigma
            max_value = mass + nsigma * sigma

        # pylint: disable=missing-kwoa
        signal = self._signal_pdf_[idx].integrate((min_value, max_value))

        signal_fracs, _, signal_err_fracs, _ = self.__get_all_fracs()

        if len(self._background_pdf_) > 0:
            frac = signal_fracs[idx]
            frac_err = signal_err_fracs[idx]
        else:
            if len(self._signal_pdf_) == 1:
                frac = 1.
                frac_err = 0.
            if idx < len(signal_fracs):
                frac = signal_fracs[idx]
                frac_err = signal_err_fracs[idx]
            else:
                frac = 1. - sum(signal_fracs)
                frac_err = np.sqrt(sum(list(err**2 for err in signal_err_fracs)))

        norm = self._data_handler_.get_norm()
        norm_err = norm * frac_err
        norm *= frac

        return float(signal * norm), float(signal * norm_err)

    def get_background(self, idx=0, **kwargs):
        """
        Get the background and its error in a given invariant-mass region

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal to be used to compute nsigma window
        **kwargs: dict
            Additional optional arguments:

            - nsigma: float
                nsigma invariant-mass window around mean for background computation

            - nhwhm: float
                number of hwhm invariant-mass window around mean for signal and background computation 
                (alternative to nsigma)

            - min: float
                minimum value of invariant-mass for background computation (alternative to nsigma)

            - max: float
                maximum value of invariant-mass for background computation (alternative to nsigma)

        Returns
        -------------------------------------------------
        background: float
            The background value obtained from the fit
        background_err: float
            The background error obtained from the fit
        """

        if not self._background_pdf_:
            Logger('Background not fitted', 'ERROR')
            return 0., 0.

        nsigma = kwargs.get('nsigma', 3.)
        nhwhm = kwargs.get('nhwhm', None)
        min_value = kwargs.get('min', None)
        max_value = kwargs.get('max', None)
        use_nsigma = True

        if nhwhm is not None and (min_value is not None or max_value is not None):
            Logger('I cannot compute the signal within a fixed mass interval and a number of HWFM', 'ERROR')
            return 0., 0.

        if min_value is not None and max_value is not None:
            use_nsigma = False

        if nhwhm is not None:
            use_nsigma = False
            if self._name_signal_pdf_[idx] not in ['gaussian', 'cauchy', 'voigtian']:
                Logger('HWHM not defined, I cannot compute the signal for this pdf', 'ERROR')
                return 0., 0.
            mass, _ = self.get_mass(idx)
            hwhm, _ = self.get_hwhm(idx)
            min_value = mass - nhwhm * hwhm
            max_value = mass + nhwhm * hwhm

        if use_nsigma:
            if self._name_signal_pdf_[idx] not in ['gaussian', 'crystalball']:
                Logger('Sigma not defined, I cannot compute the signal for this pdf', 'ERROR')
                return 0., 0.
            mass, _ = self.get_mass(idx)
            sigma, _ = self.get_sigma(idx)
            min_value = mass - nsigma * sigma
            max_value = mass + nsigma * sigma

        signal_fracs, bkg_fracs, signal_err_fracs, bkg_err_fracs = self.__get_all_fracs()

        # pylint: disable=missing-kwoa
        background, background_err = 0., 0.
        for idx2, bkg in enumerate(self._background_pdf_):

            if idx2 == len(self._background_pdf_) - 1:
                frac = 1. - sum(signal_fracs)
                frac_err = np.sqrt(sum(list(err**2 for err in signal_err_fracs)))
            else:
                frac = bkg_fracs[idx2]
                frac_err = bkg_err_fracs[idx2]

            norm = self._data_handler_.get_norm()
            norm_err = norm * frac_err
            norm *= frac

            bkg_int = bkg.integrate((min_value, max_value))
            background += bkg_int * norm
            background_err += (bkg_int * norm_err)**2

        background_err = np.sqrt(background_err)

        return float(background), float(background_err)

    def get_signal_over_background(self, idx=0, **kwargs):
        """
        Get the S/B ratio and its error in a given invariant-mass region

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal to be used to compute nsigma window
        **kwargs: dict
            Additional optional arguments:

            - nsigma: float
                nsigma invariant-mass window around mean for signal and background computation

            - nhwhm: float
                number of hwhm invariant-mass window around mean for signal and background computation 
                (alternative to nsigma)

            - min: float
                minimum value of invariant-mass for signal and background computation (alternative to nsigma)

            - max: float
                maximum value of invariant-mass for signal and background computation (alternative to nsigma)

        Returns
        -------------------------------------------------
        signal_over_background: float
            The S/B value obtained from the fit
        signal_over_background_err: float
            The S/B error obtained from the fit
        """

        signal = self.get_signal(idx, **kwargs)
        bkg = self.get_background(idx, **kwargs)
        signal_over_background = signal[0]/bkg[0]
        signal_over_background_err = np.sqrt(signal[1]**2/signal[0]**2 + bkg[1]**2/bkg[0]**2)
        signal_over_background_err *= signal_over_background

        return signal_over_background, signal_over_background_err

    def get_significance(self, idx=0, **kwargs):
        """
        Get the significance and its error in a given invariant-mass region

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal to be used to compute nsigma window
        **kwargs: dict
            Additional optional arguments:

            - nsigma: float
                nsigma invariant-mass window around mean for signal and background computation

            - nhwhm: float
                number of hwhm invariant-mass window around mean for signal and background computation 
                (alternative to nsigma)

            - min: float
                minimum value of invariant-mass for signal and background computation (alternative to nsigma)

            - max: float
                maximum value of invariant-mass for signal and background computation (alternative to nsigma)

        Returns
        -------------------------------------------------
        significance: float
            The significance value obtained from the fit
        significance_err: float
            The significance error obtained from the fit
        """

        signal = self.get_signal(idx, **kwargs)
        bkg = self.get_background(idx, **kwargs)
        significance = signal[0]/np.sqrt(signal[0]+bkg[0])
        sig_plus_bkg = signal[0] + bkg[0]

        significance_err = significance*np.sqrt(
            (signal[1]**2 + bkg[1]**2) / (4. * sig_plus_bkg**2) + (
                bkg[0]/sig_plus_bkg) * signal[1]**2 / signal[0]**2)

        return significance, significance_err

    def set_particle_mass(self, idx, **kwargs):
        """
        Set the particle mass

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal
        **kwargs: dict
            Additional optional arguments:

            - mass: float
                The mass of the particle

            - pdg_id: int
                PDG ID of the particle (alternative to mass)

            - pdg_name: str
                Name of the particle (alternative to mass)

            - limits: list
                minimum and maximum limits for the mass parameter

            - fix: bool
                fix the mass parameter
        """
        mass_name = 'm' if self._name_signal_pdf_[idx] == 'cauchy' else 'mu'
        mass = 0.
        if 'mass' in kwargs:
            mass = kwargs['mass']
        elif 'pdg_id' in kwargs:
            mass = Particle.from_pdgid(kwargs['pdg_id']).mass*1e-3
        elif 'pdg_name' in kwargs:
            mass = Particle.from_name(kwargs['pdg_name']).mass*1e-3
        else:
            Logger(f'"mass", "pdg_id", and "pdg_name" not provided, mass value for signal {idx} will not be set',
                   'ERROR')
        self._init_sgn_pars_[idx][mass_name] = mass
        if 'limits' in kwargs:
            self._limits_sgn_pars_[idx][mass_name] = kwargs['limits']
        if 'fix' in kwargs:
            self._fix_sgn_pars_[idx][mass_name] = kwargs['fix']

    def set_signal_initpar(self, idx, par_name, init_value, **kwargs):
        """
        Set a signal parameter

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal
        par_name: str
            The name of the parameter to be set
        init_value: float
            The value of parameter to be set
        **kwargs: dict
            Additional optional arguments:

            - limits: list
                minimum and maximum limits for the parameter

            - fix: bool
                fix the parameter to init_value
        """
        self._init_sgn_pars_[idx][par_name] = init_value
        if 'limits' in kwargs:
            self._limits_sgn_pars_[idx][par_name] = kwargs['limits']
        if 'fix' in kwargs:
            self._fix_sgn_pars_[idx][par_name] = kwargs['fix']

    def set_background_initpar(self, idx, par_name, init_value, **kwargs):
        """
        Set a background parameter

        Parameters
        -------------------------------------------------
        idx: int
            Index of the background
        par_name: str
            The name of the parameter to be set
        init_value: float
            The value of parameter to be set
        **kwargs: dict
            Additional optional arguments:

            - limits: list
                minimum and maximum limits for the parameter

            - fix: bool
                fix the mass parameter
        """
        self._init_bkg_pars_[idx][par_name] = init_value
        if 'limits' in kwargs:
            self._limits_bkg_pars_[idx][par_name] = kwargs['limits']
        if 'fix' in kwargs:
            self._fix_bkg_pars_[idx][par_name] = kwargs['fix']

    # pylint: disable=line-too-long
    def set_signal_kde(self, idx, sample, **kwargs):
        """
        Set sample and options for signal kde

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal
        sample: flarefly.DataHandler
            Data sample for Kernel Density Estimation
        **kwargs: dict
            Arguments for kde options. See
            https://zfit.readthedocs.io/en/latest/user_api/pdf/_generated/kde_api/zfit.pdf.KDE1DimGrid.html#zfit.pdf.KDE1DimGrid
            for more details
        """

        self._kde_signal_sample_[idx] = sample
        self._kde_signal_option_[idx] = kwargs

    # pylint: disable=line-too-long
    def set_background_kde(self, idx, sample, **kwargs):
        """
        Set sample and options for background kde

        Parameters
        -------------------------------------------------
        idx: int
            Index of the background
        sample: flarefly.DataHandler
            Data sample for Kernel Density Estimation
        **kwargs: dict
            Arguments for kde options. See
            https://zfit.readthedocs.io/en/latest/user_api/pdf/_generated/kde_api/zfit.pdf.KDE1DimGrid.html#zfit.pdf.KDE1DimGrid
            for more details
        """

        self._kde_bkg_sample_[idx] = sample
        self._kde_bkg_option_[idx] = kwargs

    def __write_data(self, hdata, histname='hdata', filename='output.root', option='recreate'):
        """
        Helper method to save a data histogram in a .root file (TH1D format)

        Parameters
        -------------------------------------------------
        hdata: hist
            Histogram containing the data

        histname: str
            Name of the histogram

        filename: str
            Name of the ROOT file

        option: str
            Option (recreate or update)
        """
        if option not in ['recreate', 'update']:
            Logger('Illegal option to save outputs in ROOT file!', 'FATAL')

        if option == 'recreate':
            with uproot.recreate(filename) as ofile:
                ofile[histname] = hdata
        else:
            with uproot.update(filename) as ofile:
                ofile[histname] = hdata

    def __write_pdf(self, histname, weight, num, filename='output.root', option='recreate'):
        """
        Helper method to save a pdf histogram in a .root file (TH1D format)

        Parameters
        -------------------------------------------------
        histname: str
            Name of the histogram

        weight: array[float]
            Array of weights for histogram bins

        num: int
            Number of bins to plot pdfs converted into histograms

        filename: str
            ROOT file name

        option: str
            Option (recreate or update)
        """

        if option not in ['recreate', 'update']:
            Logger('Illegal option to save outputs in ROOT file!', 'FATAL')

        limits = self._data_handler_.get_limits()
        x_plot = np.linspace(limits[0], limits[1], num=num)
        histo = Hist.new.Reg(num, limits[0], limits[1], name=self._data_handler_.get_var_name()).Double()
        histo.fill(x_plot, weight=weight)

        if option == 'recreate':
            with uproot.recreate(filename) as ofile:
                ofile[histname] = histo
        else:
            with uproot.update(filename) as ofile:
                ofile[histname] = histo
