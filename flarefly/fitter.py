"""
Module containing the class used to perform mass fits
"""
import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"  # pylint: disable=wrong-import-position
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnchoredText

import zfit
import uproot
from hist import Hist
import mplhep
from hepstats.splot import compute_sweights
import pdg
from flarefly.utils import Logger
from flarefly.pdf_builder import PDFBuilder
import flarefly.custom_pdfs as cpdf


# pylint: disable=too-many-instance-attributes, too-many-lines, too-many-public-methods
class F2MassFitter:
    """
    Class used to perform mass fits with the zfit library
    https://github.com/zfit/zfit
    """

    # pylint: disable=too-many-statements, too-many-branches
    def __init__(self, data_handler, name_signal_pdf, name_background_pdf, **kwargs):
        """
        Initialize the F2MassFitter class
        Parameters
        -------------------------------------------------
        data_handler: flarefly.DataHandler
            The data handler containing the data to fit

        name_signal_pdf: list
            The list of names for the signal pdfs. The possible options are:

            - 'nosignal'

            - 'gaussian'

            - 'doublegaus'

            - 'gausexptail'

            - 'genergausexptail'

            - 'genergausexptailsymm'

            - 'bifurgaus'

            - 'crystalball'

            - 'doublecb'

            - 'doublecbsymm'

            - 'genercrystalball'

            - 'cauchy'

            - 'voigtian'

            - 'kde_exact' (requires to set the datasample and options)

            - 'kde_grid' (requires to set the datasample and options)

            - 'kde_fft' (requires to set the datasample and options)

            - 'kde_isj' (requires to set the datasample and options)

            - 'hist' (only for binned fits, requires to set the datasample)

        name_background_pdf: list
            The list of names of the background pdfs. The possible options are:

            - 'nobkg'

            - 'expo'

            - 'powlaw'

            - 'expopow'

            - 'expopowext'

            - 'chebpolN' (N is the order of the polynomial)

            - 'kde_exact' (requires to set the datasample and options)

            - 'kde_grid' (requires to set the datasample and options)

            - 'kde_fft' (requires to set the datasample and options)

            - 'kde_isj' (requires to set the datasample and options)

            - 'hist' (only for binned fits, requires to set the datasample)

        **kwargs: dict
            Additional optional arguments:

            - name_refl_pdf: list
                The list of names of the signal pdfs. It must have the same length as the signal list.
                The possible options are:

                - 'kde_exact' (requires to set the datasample and options)

                - 'kde_grid' (requires to set the datasample and options)

                - 'kde_fft' (requires to set the datasample and options)

                - 'kde_isj' (requires to set the datasample and options)

                - 'hist' (only for binned fits, requires to set the datasample)

            - name: str
                Optional name for the fitter,
                needed in case of multiple fitters defined in the same script

            - chi2_loss: bool
                chi2 minimization if True, nll minmization else,
                default value to False

            - minuit_mode:
                A number used by minuit to define the internal minimization strategy, either 0, 1 or 2.
                0 is the fastest, 2 is the slowest
                (see more details in
                https://zfit.readthedocs.io/en/latest/user_api/minimize/_generated/minimizers/zfit.minimize.Minuit.html#zfit.minimize.Minuit)
                Default value to 0

            - limits: list
                list of fit limits to include in the fit

            - tol: float
                Termination value for the convergence/stopping criterion of the algorithm in order to determine
                if the minimum has been found.
                Default value to 0.001

            - verbosity: int
                verbosity level (from 0 to 10)
                Default value to 0

            - signal_at_threshold: list
                list of booleans which indicate whether the signal PDFs are at threshold or not.
                Each element corresponds to a signal pdf

            - label_signal_pdf: list
                list of labels for signal pdfs

            - label_bkg_pdf: list
                list of labels for background pdfs
        """

        self._data_handler_ = data_handler
        self._name_signal_pdf_ = name_signal_pdf
        self._name_background_pdf_ = name_background_pdf
        self.label_signal_pdf = kwargs.get(
            'label_signal_pdf', [f'signal {idx}' for idx in range(len(name_signal_pdf))])
        self.label_bkg_pdf = kwargs.get(
            'label_bkg_pdf', [f'background {idx}' for idx in range(len(name_background_pdf))])
        self._name_refl_pdf_ = kwargs.get('name_refl_pdf', [None for _ in name_signal_pdf])
        if len(self._name_refl_pdf_) != len(self._name_signal_pdf_):
            Logger('List of pdfs for signals and reflections different! Exit', 'FATAL')
        if self._name_signal_pdf_[0] == 'nosignal':
            self._signal_pdf_ = []
            self._hist_signal_sample_ = []
            self._kde_signal_sample_ = []
            self._kde_signal_option_ = []
            self._name_refl_pdf_ = []
        else:
            self._signal_pdf_ = [None for _ in name_signal_pdf]
            self._hist_signal_sample_ = [None for _ in name_signal_pdf]
            self._kde_signal_sample_ = [None for _ in name_signal_pdf]
            self._kde_signal_option_ = [None for _ in name_signal_pdf]
        if self._name_background_pdf_[0] == 'nobkg':
            self._background_pdf_ = []
            self._hist_bkg_sample_ = []
            self._kde_bkg_sample_ = []
            self._kde_bkg_option_ = []
        else:
            self._background_pdf_ = [None for _ in name_background_pdf]
            self._hist_bkg_sample_ = [None for _ in name_background_pdf]
            self._kde_bkg_sample_ = [None for _ in name_background_pdf]
            self._kde_bkg_option_ = [None for _ in name_background_pdf]

        self._refl_pdf_ = [None for _ in self._name_refl_pdf_]
        self._hist_refl_sample_ = [None for _ in self._name_refl_pdf_]
        self._kde_refl_sample_ = [None for _ in self._name_refl_pdf_]
        self._kde_refl_option_ = [None for _ in self._name_refl_pdf_]
        self._refl_over_sgn_ = [0. for _ in self._name_refl_pdf_]
        self._total_pdf_ = None
        self._total_pdf_binned_ = None
        self._fit_result_ = None
        self._init_sgn_pars_ = [{} for _ in name_signal_pdf]
        self._init_bkg_pars_ = [{} for _ in name_background_pdf]
        self._limits_sgn_pars_ = [{} for _ in name_signal_pdf]
        self._limits_bkg_pars_ = [{} for _ in name_background_pdf]
        self._fix_sgn_pars_ = [{} for _ in name_signal_pdf]
        self._fix_fracs_to_pdfs_ = []
        self._fix_bkg_pars_ = [{} for _ in name_background_pdf]
        self._sgn_pars_ = [{} for _ in name_signal_pdf]
        self._bkg_pars_ = [{} for _ in name_background_pdf]
        if self._name_signal_pdf_[0] != 'nosignal' and self._name_background_pdf_[0] == 'nobkg':
            if self._name_refl_pdf_[0] is not None:
                Logger('Not possible to use reflections without background pdf', 'FATAL')
            self._fracs_ = [None for _ in range(len(name_signal_pdf) - 1)]
        elif self._name_signal_pdf_[0] == 'nosignal' and self._name_background_pdf_[0] != 'nobkg':
            self._fracs_ = [None for _ in range(len(name_background_pdf) - 1)]
        elif self._name_signal_pdf_[0] != 'nosignal' and self._name_background_pdf_[0] != 'nobkg':
            self._fracs_ = [None for _ in range(2 * len(name_signal_pdf) + len(name_background_pdf) - 1)]
        else:
            Logger('No signal nor background pdf defined', 'FATAL')
        self._total_pdf_norm_ = None
        self._ratio_truncated_ = None
        self._rawyield_ = [0. for _ in name_signal_pdf]
        self._rawyield_err_ = [0. for _ in name_signal_pdf]
        self._minimizer_ = zfit.minimize.Minuit(
            verbosity=kwargs.get('verbosity', 7),
            mode=kwargs.get('minuit_mode', 0),
            tol=kwargs.get('tol', 0.001)
        )
        if 'limits' in kwargs and self._data_handler_.get_is_binned():
            Logger('Restriction of fit limits is not yet implemented in binned fits!', 'FATAL')
        if 'limits' in kwargs and 'nosignal' in self._name_signal_pdf_:
            Logger('Using signal PDFs while restricting fit limits!', 'WARNING')
        self._limits_ = kwargs.get(
            'limits', self._data_handler_.get_limits())
        if not isinstance(self._limits_[0], list):
            self._limits_ = [self._limits_]
        self._is_truncated_ = self._limits_ != [self._data_handler_.get_limits()]
        self._name_ = kwargs.get('name', 'fitter')
        self._ndf_ = None
        self._chi2_loss_ = kwargs.get('chi2_loss', False)
        self._base_sgn_cmap_ = plt.colormaps.get_cmap('viridis')
        self._sgn_cmap_ = ListedColormap(self._base_sgn_cmap_(np.linspace(0.4, 0.65, len(self._signal_pdf_))))
        n_bkg_colors = len(self._background_pdf_)
        if len(self._background_pdf_) == 0:
            n_bkg_colors = 1  # to avoid crash
        self._base_bkg_cmap_ = plt.colormaps.get_cmap('Reds')
        self._bkg_cmap_ = ListedColormap(self._base_bkg_cmap_(np.linspace(0.8, 0.2, n_bkg_colors)))
        self._base_refl_cmap_ = plt.colormaps.get_cmap('summer')
        self._refl_cmap_ = ListedColormap(self._base_refl_cmap_(np.linspace(0., 0.6, len(self._refl_pdf_))))

        self._raw_residuals_ = []
        self._raw_residual_variances_ = []
        self._std_residuals_ = []
        self._std_residual_variances_ = []

        self._signal_at_threshold = kwargs.get('signal_at_threshold', [False for _ in name_signal_pdf])
        self._signalthr_pdf_ = [None for _ in name_signal_pdf]

        zfit.settings.advanced_warnings.all = False
        zfit.settings.changed_warnings.all = False
        self.pdg_api = pdg.connect()

    def cleanup(self):
        """
        Cleanup function to clear the graph cache of zfit
        """
        zfit.run.clear_graph_cache()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    # pylint: disable=too-many-branches, too-many-statements
    def __build_signal_pdfs(self, obs):
        """
        Helper function to compose the signal pdfs
        """

        for ipdf, (pdf_name, at_threshold) in enumerate(zip(self._name_signal_pdf_, self._signal_at_threshold)):
            if pdf_name == 'nosignal':
                Logger('Performing fit with no signal pdf', 'WARNING')
                break
            if 'kde' in pdf_name:
                self._signal_pdf_[ipdf] = PDFBuilder.build_signal_kde(
                    pdf_name,
                    self._kde_signal_sample_[ipdf],
                    self._name_,
                    ipdf,
                    self._kde_signal_option_[ipdf]
                )
            elif pdf_name == 'hist':
                self._signal_pdf_[ipdf] = PDFBuilder.build_signal_hist(
                    obs,
                    self._hist_signal_sample_[ipdf],
                    self._name_,
                    ipdf,
                )
            else:
                self._signal_pdf_[ipdf], self._sgn_pars_[ipdf] = PDFBuilder.build_signal_pdf(
                    pdf_name,
                    obs,
                    self._name_,
                    ipdf,
                    self._init_sgn_pars_[ipdf],
                    self._limits_sgn_pars_[ipdf],
                    self._fix_sgn_pars_[ipdf]
                )

            if at_threshold:
                # pion mass as default
                self._init_sgn_pars_[ipdf].setdefault('powerthr', 1.)
                self._limits_sgn_pars_[ipdf].setdefault('massthr', [None, None])
                self._fix_sgn_pars_[ipdf].setdefault('powerthr', False)
                self._fix_sgn_pars_[ipdf].setdefault('massthr', True)
                self._limits_sgn_pars_[ipdf].setdefault('powerthr', [None, None])
                self._init_sgn_pars_[ipdf].setdefault('massthr', self.pdg_api.get_particle_by_mcid(211).mass)
                self._sgn_pars_[ipdf][f'{self._name_}_massthr_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_massthr_signal{ipdf}', self._init_sgn_pars_[ipdf]['massthr'],
                    self._limits_sgn_pars_[ipdf]['massthr'][0], self._limits_sgn_pars_[ipdf]['massthr'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['massthr'])
                self._sgn_pars_[ipdf][f'{self._name_}_powerthr_signal{ipdf}'] = zfit.Parameter(
                    f'{self._name_}_powerthr_signal{ipdf}', self._init_sgn_pars_[ipdf]['powerthr'],
                    self._limits_sgn_pars_[ipdf]['powerthr'][0], self._limits_sgn_pars_[ipdf]['powerthr'][1],
                    floating=not self._fix_sgn_pars_[ipdf]['powerthr'])
                self._signalthr_pdf_[ipdf] = cpdf.Pow(
                    obs=obs,
                    mass=self._sgn_pars_[ipdf][f'{self._name_}_massthr_signal{ipdf}'],
                    power=self._sgn_pars_[ipdf][f'{self._name_}_powerthr_signal{ipdf}']
                )
                self._signal_pdf_[ipdf] = zfit.pdf.ProductPDF(
                    [self._signal_pdf_[ipdf], self._signalthr_pdf_[ipdf]], obs=obs)

    def __build_background_pdfs(self, obs):
        """
        Helper function to compose the background pdfs
        """

        for ipdf, pdf_name in enumerate(self._name_background_pdf_):
            if pdf_name == 'nobkg':
                Logger('Performing fit with no bkg pdf', 'WARNING')
                break
            if 'kde' in pdf_name:
                self._background_pdf_[ipdf] = PDFBuilder.build_bkg_kde(
                    pdf_name,
                    self._kde_bkg_sample_[ipdf],
                    self._name_,
                    ipdf,
                    self._kde_bkg_option_[ipdf]
                )

            elif pdf_name == 'hist':
                self._background_pdf_[ipdf] = PDFBuilder.build_bkg_hist(
                    obs,
                    self._hist_bkg_sample_[ipdf],
                    self._name_,
                    ipdf,
                )
            else:
                self._background_pdf_[ipdf], self._bkg_pars_[ipdf] = PDFBuilder.build_bkg_pdf(
                    pdf_name,
                    obs,
                    self._name_,
                    ipdf,
                    self._init_bkg_pars_[ipdf],
                    self._limits_bkg_pars_[ipdf],
                    self._fix_bkg_pars_[ipdf]
                )
                if pdf_name in ['powlaw', 'expopow', 'expopowext'] and\
                        self._data_handler_.get_limits()[0] < self._init_bkg_pars_[ipdf]["mass"]:
                    Logger(
                        'The mass parameter in powlaw cannot be smaller than the lower fit limit, '
                        'please fix it.',
                        'FATAL'
                    )

    # pylint: disable=too-many-branches, too-many-statements
    def __build_reflection_pdfs(self, obs):
        """
        Helper function to compose the reflection pdfs
        """

        for ipdf, pdf_name in enumerate(self._name_refl_pdf_):
            if pdf_name is None:  # by default we put a dummy pdf
                low = zfit.Parameter(f'{self._name_}_low_refl{ipdf}',
                                     self._data_handler_.get_limits()[0],
                                     self._data_handler_.get_limits()[0] - 0.01,
                                     self._data_handler_.get_limits()[0] + 0.01,
                                     floating=False)
                high = zfit.Parameter(f'{self._name_}_high_refl{ipdf}',
                                      self._data_handler_.get_limits()[1],
                                      self._data_handler_.get_limits()[1] - 0.01,
                                      self._data_handler_.get_limits()[1] + 0.01,
                                      floating=False)
                self._refl_pdf_[ipdf] = zfit.pdf.Uniform(obs=obs, low=low, high=high)
            elif 'kde' in pdf_name:
                if self._kde_refl_sample_[ipdf]:
                    if pdf_name == 'kde_exact':
                        self._refl_pdf_[ipdf] = zfit.pdf.KDE1DimExact(self._kde_refl_sample_[ipdf].get_data(),
                                                                      obs=self._kde_refl_sample_[ipdf].get_obs(),
                                                                      name=f'{self._name_}_kde_refl{ipdf}',
                                                                      **self._kde_refl_option_[ipdf])
                    elif pdf_name == 'kde_grid':
                        self._refl_pdf_[ipdf] = zfit.pdf.KDE1DimGrid(self._kde_refl_sample_[ipdf].get_data(),
                                                                     obs=self._kde_refl_sample_[ipdf].get_obs(),
                                                                     name=f'{self._name_}_kde_refl{ipdf}',
                                                                     **self._kde_refl_option_[ipdf])
                    elif pdf_name == 'kde_fft':
                        self._refl_pdf_[ipdf] = zfit.pdf.KDE1DimFFT(self._kde_refl_sample_[ipdf].get_data(),
                                                                    obs=self._kde_refl_sample_[ipdf].get_obs(),
                                                                    name=f'{self._name_}_kde_refl{ipdf}',
                                                                    **self._kde_refl_option_[ipdf])
                    elif pdf_name == 'kde_isj':
                        self._refl_pdf_[ipdf] = zfit.pdf.KDE1DimISJ(self._kde_refl_sample_[ipdf].get_data(),
                                                                    obs=self._kde_refl_sample_[ipdf].get_obs(),
                                                                    name=f'{self._name_}_kde_refl{ipdf}',
                                                                    **self._kde_refl_option_[ipdf])
                else:
                    Logger(f'Missing datasample for Kernel Density Estimation of reflection {ipdf}!', 'FATAL')
            elif pdf_name == 'hist':
                if self._hist_refl_sample_[ipdf]:
                    self._refl_pdf_[ipdf] = zfit.pdf.SplinePDF(
                        zfit.pdf.HistogramPDF(self._hist_refl_sample_[ipdf].get_binned_data(),
                                              name=f'{self._name_}_hist_refl{ipdf}'),
                        order=3,
                        obs=obs
                    )
                else:
                    Logger(f'Missing datasample for histogram template of reflections {ipdf}!', 'FATAL')
            else:
                Logger(f'Reflection pdf {pdf_name} not supported', 'FATAL')

    def __get_constrained_frac_par(self, frac_par, factor_par, refl=False):
        """
        Helper function to create a parameter constrained to a value
        """

        def par_func(par, factor):
            return par * factor

        if refl:
            name_composed_param = f'{self._name_}_{factor_par.name.replace("factor", "frac_refl")}'
        else:
            name_composed_param = f'{self._name_}_{factor_par.name.replace("factor", "frac")}'

        frac_par_constrained = zfit.ComposedParameter(name_composed_param,
                                                      par_func,
                                                      params=[frac_par, factor_par],
                                                      unpack_params=True
                                                      )
        return frac_par_constrained

    def __set_frac_constraints(self):
        for frac_fix_info in self._fix_fracs_to_pdfs_:
            if frac_fix_info['fixed_pdf_type'] == 'signal':
                if frac_fix_info['target_pdf_type'] == 'signal':
                    self._fracs_[frac_fix_info['fixed_pdf_idx']] = self.__get_constrained_frac_par(
                        self._fracs_[frac_fix_info['target_pdf_idx']],
                        frac_fix_info['factor']
                    )
                    # also set for reflections
                    fix_factor = zfit.Parameter(
                        frac_fix_info['factor'].name.replace('factor', 'factor_refl'),
                        float(
                            frac_fix_info['factor'].value()
                        ) * self._refl_over_sgn_[frac_fix_info['fixed_pdf_idx']],
                        floating=False
                    )
                    self._fracs_[frac_fix_info['fixed_pdf_idx'] + len(self._signal_pdf_)] = \
                        self.__get_constrained_frac_par(
                            self._fracs_[frac_fix_info['target_pdf_idx']],
                            fix_factor,
                            refl=True
                        )
                else:  # fix to background PDF
                    self._fracs_[frac_fix_info['fixed_pdf_idx']] = self.__get_constrained_frac_par(
                        self._fracs_[frac_fix_info['target_pdf_idx'] + 2 * len(self._signal_pdf_)],
                        frac_fix_info['factor']
                    )
                    # also set for reflections
                    fix_factor = zfit.Parameter(
                        frac_fix_info['factor'].name.replace('factor', 'factor_refl'),
                        float(
                            frac_fix_info['factor'].value()
                        ) * self._refl_over_sgn_[frac_fix_info['fixed_pdf_idx']],
                        floating=False
                    )
                    self._fracs_[frac_fix_info['fixed_pdf_idx'] + len(self._signal_pdf_)] = \
                        self.__get_constrained_frac_par(
                            self._fracs_[frac_fix_info['target_pdf_idx'] + 2 * len(self._signal_pdf_)],
                            fix_factor,
                            refl=True
                        )
            else:  # fix background fraction
                if frac_fix_info['target_pdf_type'] == 'signal':
                    self._fracs_[frac_fix_info['fixed_pdf_idx'] + 2 * len(self._signal_pdf_)] = \
                        self.__get_constrained_frac_par(
                            self._fracs_[frac_fix_info['target_pdf_idx']],
                            frac_fix_info['factor']
                        )
                else:  # fix to background PDF
                    self._fracs_[frac_fix_info['fixed_pdf_idx'] + 2 * len(self._signal_pdf_)] = \
                        self.__get_constrained_frac_par(
                            self._fracs_[frac_fix_info['target_pdf_idx'] + 2 * len(self._signal_pdf_)],
                            frac_fix_info['factor']
                        )

    def __build_total_pdf(self):
        """
        Helper function to compose the total pdf
        """

        if self._data_handler_.get_is_binned():  # we need unbinned pdfs to sum them
            obs = self._data_handler_.get_unbinned_obs_from_binned_data()
        else:
            obs = self._data_handler_.get_obs()

        # order of the pdfs is signal, background

        self.__build_signal_pdfs(obs)
        self.__build_background_pdfs(obs)
        if len(self._background_pdf_) != 0:
            self.__build_reflection_pdfs(obs)
        else:
            self._refl_pdf_ = []

        self.__get_total_pdf_norm()

        if len(self._signal_pdf_) + len(self._background_pdf_) == 1:
            if len(self._signal_pdf_) == 0:
                self._total_pdf_ = self._background_pdf_[0]
                if self._is_truncated_:
                    self._total_pdf_ = self._total_pdf_.to_truncated(limits=self._limits_, obs=obs)
                return
            if len(self._background_pdf_) == 0:
                self._total_pdf_ = self._signal_pdf_[0]
                if self._is_truncated_:
                    self._total_pdf_ = self._total_pdf_.to_truncated(limits=self._limits_, obs=obs)
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

            # normalisation of reflection fixed to the one of the signal
            if len(self._background_pdf_) != 0:
                def func_mult(params):
                    return params['ros'] * params['s']
                self._fracs_[ipdf + len(self._signal_pdf_)] = zfit.ComposedParameter(
                    f'{self._name_}_frac_refl{ipdf}',
                    func_mult, params={'ros': self._refl_over_sgn_[ipdf],
                                       's': self._fracs_[ipdf]}
                )

        if len(self._background_pdf_) > 1:
            for ipdf, _ in enumerate(self._background_pdf_[:-1]):
                self._init_bkg_pars_[ipdf].setdefault('frac', 0.1)
                self._fix_bkg_pars_[ipdf].setdefault('frac', False)
                self._limits_bkg_pars_[ipdf].setdefault('frac', [0, 1.])
                self._fracs_[ipdf + 2 * len(self._signal_pdf_)] = zfit.Parameter(
                    f'{self._name_}_frac_bkg{ipdf}',
                    self._init_bkg_pars_[ipdf]['frac'],
                    self._limits_bkg_pars_[ipdf]['frac'][0],
                    self._limits_bkg_pars_[ipdf]['frac'][1],
                    floating=not self._fix_bkg_pars_[ipdf]['frac'])

        self.__set_frac_constraints()

        self._total_pdf_ = zfit.pdf.SumPDF(self._signal_pdf_+self._refl_pdf_+self._background_pdf_,
                                           self._fracs_)

        if not self._data_handler_.get_is_binned() and self._is_truncated_:
            self._total_pdf_ = self._total_pdf_.to_truncated(limits=self._limits_, obs=obs, norm=obs)

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

        if len(self._signal_pdf_) + len(self._background_pdf_) == 1:
            if len(self._signal_pdf_) == 0:
                self._total_pdf_binned_ = zfit.pdf.BinnedFromUnbinnedPDF(self._background_pdf_[0], obs)
                return
            if len(self._background_pdf_) == 0:
                self._total_pdf_binned_ = zfit.pdf.BinnedFromUnbinnedPDF(self._signal_pdf_[0], obs)
                return

        self._total_pdf_binned_ = zfit.pdf.BinnedFromUnbinnedPDF(zfit.pdf.SumPDF(
            self._signal_pdf_+self._refl_pdf_+self._background_pdf_, self._fracs_), obs)

    def __get_frac_and_err(self, par_name, frac_par, frac_type):
        if 'constrained' in par_name:
            split_par_name = par_name.split(sep='_constrained_to_')
            target_frac_name = split_par_name[0].replace(split_par_name[0].split('_')[-1], split_par_name[1])
            # find the parameter containing target_frac_name
            target_par_name, target_par = [
                (par.name, par) for par in self._fracs_ if target_frac_name in par.name
            ][0]

            if 'constrained' in target_par_name:
                if f'{self._name_}_frac_signal' in target_par_name:
                    frac_type = 'signal'
                else:
                    frac_type = 'bkg'
                frac_temp, err_temp, isgn = self.__get_frac_and_err(target_par_name, target_par, frac_type)
                return (
                    frac_temp * float(frac_par.params['param_1'].value()),
                    err_temp * float(frac_par.params['param_1'].value()),
                    isgn
                )
            frac = self._fit_result_.params[target_frac_name]['value']
            frac_err = self._fit_result_.params[target_frac_name]['hesse']['error']

            return (
                frac * float(frac_par.params['param_1'].value()),
                frac_err * float(frac_par.params['param_1'].value()),
                int(split_par_name[0][-1])
            )

        return (
            self._fit_result_.params[par_name]['value'],
            self._fit_result_.params[par_name]['hesse']['error'],
            int(par_name.split(sep=f'{self._name_}_frac_{frac_type}')[-1])
        )

    def __get_all_fracs(self):
        """
        Helper function to get all fractions

        Returns
        -------------------------------------------------
        signal_fracs: list
            fractions of the signal pdfs
        background_fracs: list
            fractions of the background pdfs
        refl_fracs: list
            fractions of the reflected signal pdfs
        signal_err_fracs: list
            errors of fractions of the signal pdfs
        bkg_err_fracs: list
            errors of fractions of the background pdfs
        refl_err_fracs: list
            errors of fractions of the reflected signal pdfs
        """
        signal_fracs, bkg_fracs, refl_fracs, signal_err_fracs, bkg_err_fracs, refl_err_fracs = ([] for _ in range(6))
        for frac_par in self._fracs_:
            if frac_par is None:
                continue
            par_name = frac_par.name
            if f'{self._name_}_frac_signal' in par_name:
                signal_frac, signal_err, isgn = self.__get_frac_and_err(par_name, frac_par, 'signal')
                signal_fracs.append(signal_frac)
                signal_err_fracs.append(signal_err)
                refl_fracs.append(signal_frac * self._refl_over_sgn_[isgn])
                refl_err_fracs.append(signal_err * self._refl_over_sgn_[isgn])
            elif f'{self._name_}_frac_bkg' in par_name:
                bkg_frac, bkg_err, isgn = self.__get_frac_and_err(par_name, frac_par, 'bkg')
                bkg_fracs.append(bkg_frac)
                bkg_err_fracs.append(bkg_err)

        if len(signal_fracs) == len(bkg_fracs) == len(refl_fracs) == 0:
            if len(self._background_pdf_) == 0:
                signal_fracs.append(1.)
                signal_err_fracs.append(0.)
                refl_fracs.append(0.)
                refl_err_fracs.append(0.)
            elif len(self._signal_pdf_) == 0:
                bkg_fracs.append(1.)
                bkg_err_fracs.append(0.)
        else:
            if len(self._background_pdf_) == 0:
                signal_fracs.append(1 - sum(signal_fracs) - sum(refl_fracs) - sum(bkg_fracs))
                signal_err_fracs.append(np.sqrt(sum(list(err**2 for err in signal_err_fracs + bkg_err_fracs))))
            else:
                bkg_fracs.append(1 - sum(signal_fracs) - sum(refl_fracs) - sum(bkg_fracs))
                bkg_err_fracs.append(np.sqrt(sum(list(err**2 for err in signal_err_fracs + bkg_err_fracs))))

        return signal_fracs, bkg_fracs, refl_fracs, signal_err_fracs, bkg_err_fracs, refl_err_fracs

    def __get_total_pdf_norm(self):
        """
        Get the normalization of the total pdf
        """
        if self._data_handler_.get_is_binned():
            self._total_pdf_norm_ = float(self._data_handler_.get_norm())
            return

        norm_total_pdf = 0.
        for lim in self._limits_:
            norm_obs = self._data_handler_.get_obs().with_limits(lim)
            norm_total_pdf += float(self._data_handler_.get_data().with_obs(norm_obs).n_events)
        self._total_pdf_norm_ = norm_total_pdf

    def __get_ratio_truncated(self):
        """
        Get the ratio of the sidebands integral over the total integral
        """
        if not self._is_truncated_:
            self._ratio_truncated_ = 1.
            return

        signal_fracs, bkg_fracs, refl_fracs, _, _, _ = self.__get_all_fracs()
        fracs = signal_fracs + refl_fracs + bkg_fracs
        limits = self._data_handler_.get_limits()
        sidebands_integral, total_integral = 0., 0.
        for i_pdf, pdf in enumerate(self._signal_pdf_ + self._refl_pdf_ + self._background_pdf_):
            sidebands_integral += float(sum(pdf.integrate(lim) * fracs[i_pdf] for lim in self._limits_))
            total_integral += float(pdf.integrate(limits) * fracs[i_pdf])

        self._ratio_truncated_ = sidebands_integral / total_integral

    def __get_raw_residuals(self):
        """
        Get the raw residuals (data_value - bkg_model_value) for all bins
        """

        bins = self._data_handler_.get_nbins()
        norm = self._total_pdf_norm_
        self._raw_residuals_ = [None]*bins
        background_pdf_binned_ = [None for _ in enumerate(self._name_background_pdf_)]
        model_bkg_values = [None for _ in enumerate(self._name_background_pdf_)]

        # access normalized data values and errors for all bins
        if self._data_handler_.get_is_binned():
            binned_data = self._data_handler_.get_binned_data()
            data_values = binned_data.values()
            self._raw_residual_variances_ = binned_data.variances()
            obs = self._data_handler_.get_obs()
        else:
            data_values = self._data_handler_.get_binned_data_from_unbinned_data()
            self._raw_residual_variances_ = data_values  # poissonian errors
            obs = self._data_handler_.get_binned_obs_from_unbinned_data()

        # get background fractions
        if len(self._background_pdf_) > 0:
            if len(self._background_pdf_) == 1:
                _, bkg_fracs, _, _, _, _ = self.__get_all_fracs()
            else:
                _, bkg_fracs, _, _, _, _ = self.__get_all_fracs()
            # access model predicted values for background
            for ipdf, bkg_name in enumerate(self._name_background_pdf_):
                background_pdf_binned_[ipdf] = zfit.pdf.BinnedFromUnbinnedPDF(self._background_pdf_[ipdf], obs)
                norm_bkg = norm
                if "hist" in bkg_name:
                    norm_bkg /= float(sum(background_pdf_binned_[ipdf].values()))
                model_bkg_values[ipdf] = background_pdf_binned_[ipdf].values() * bkg_fracs[ipdf]
                model_bkg_values[ipdf] *= norm_bkg / self._ratio_truncated_
            # compute residuals
            for ibin, data in enumerate(data_values):
                self._raw_residuals_[ibin] = float(data)
                for ipdf, _ in enumerate(self._name_background_pdf_):
                    self._raw_residuals_[ibin] -= model_bkg_values[ipdf][ibin]
        else:
            for ibin, data in enumerate(data_values):
                self._raw_residuals_[ibin] = float(data)

    def __get_std_residuals(self):
        """
        Get the standardized residuals
        (data_value - bkg_model_value)/ sigma_data for all bins
        """

        bins = self._data_handler_.get_nbins()
        norm = self._total_pdf_norm_
        self._std_residuals_ = [None]*bins
        self._std_residual_variances_ = [None]*bins

        # access normalized data values and errors for all bins
        if self._data_handler_.get_is_binned():
            binned_data = self._data_handler_.get_binned_data()
            data_values = binned_data.values()
            variances = binned_data.variances()
        else:
            if self._limits_ != [self._data_handler_.get_limits()]:  # restricted fit limits
                Logger('Standard residuals not yet implemented with restricted fit limits', 'FATAL')
            data_values = self._data_handler_.get_binned_data_from_unbinned_data()
            variances = data_values  # poissonian errors

        # access model predicted values for background
        self.__build_total_pdf_binned()
        model_values = self._total_pdf_binned_.values()*norm/self._ratio_truncated_
        for ibin, (data, model, variance) in enumerate(zip(data_values, model_values, variances)):
            if variance == 0:
                Logger('Null variance. Consider enlarging the bins.', 'FATAL')
            self._std_residuals_[ibin] = float((data - model)/np.sqrt(variance))
            self._std_residual_variances_[ibin] = float(variance/np.sqrt(variance))

    def __prefit(self, excluded_regions):
        """
        Perform a prefit to the background distributions with the zfit library.
        The expected signal regions must be provided.

        Parameters
        -------------------------------------------------
        excluded_regions: list
            List or list of lists with limits for the excluded regions

        """

        if self._name_background_pdf_[0] == "nobkg":
            Logger('Prefit cannot be performed in case of no background, skip', 'WARNING')
            return

        if self._data_handler_.get_is_binned():
            Logger('Prefit not yet implemented for binned fits, skip', 'WARNING')
            return

        if len(self._limits_) > 1:
            Logger('Prefit not supported for fits on disjoint intervals, skip', 'WARNING')
            return

        obs = self._data_handler_.get_obs()

        # we build the limits
        if not isinstance(excluded_regions[0], list):
            excluded_regions = [excluded_regions]

        # check if excluded regions overlap
        limits_sb = []
        if excluded_regions[0][0] > self._limits_[0][0]:
            limits_sb.append([self._limits_[0][0], excluded_regions[0][0]])
        for iregion, region in enumerate(excluded_regions[:-1]):
            if region[1] > self._limits_[0][0] and \
                excluded_regions[iregion+1][0] < self._limits_[-1][-1] and \
                    region[1] < excluded_regions[iregion+1][0]:
                limits_sb.append([region[1], excluded_regions[iregion+1][0]])
        if excluded_regions[-1][1] < self._limits_[-1][-1]:
            limits_sb.append([excluded_regions[-1][1], self._limits_[-1][-1]])

        if len(limits_sb) == 0:
            Logger('Cannot perform prefit with the excluded regions set since no sidebands left, skip', 'WARNING')
            return

        fracs_bkg_prefit = []
        for ipdf, _ in enumerate(self._background_pdf_):
            fracs_bkg_prefit.append(zfit.Parameter(f'{self._name_}_frac_bkg{ipdf}_prefit', 0.1, 0., 1.))

        prefit_background_pdf = None
        if len(self._background_pdf_) > 1:
            prefit_background_pdf = zfit.pdf.SumPDF(self._background_pdf_, fracs_bkg_prefit).to_truncated(
                limits=limits_sb, obs=obs, norm=obs)
        else:
            prefit_background_pdf = self._background_pdf_[0].to_truncated(
                limits=limits_sb, obs=obs, norm=obs)

        prefit_loss = zfit.loss.UnbinnedNLL(model=prefit_background_pdf,
                                            data=self._data_handler_.get_data())

        Logger("Performing background only PREFIT", "INFO")
        res_prefit = self._minimizer_.minimize(loss=prefit_loss)
        Logger(res_prefit, 'RESULT')

        # re-initialise the parameters from those obtained in the prefit
        for par in res_prefit.params:
            which_pdf = int(par.name[-1])
            self._bkg_pars_[which_pdf][par.name].set_value(res_prefit.params[par.name]['value'])

    # pylint: disable=too-many-nested-blocks
    def mass_zfit(self, do_prefit=False, **kwargs):
        """
        Perform a mass fit with the zfit library

        Parameters
        -------------------------------------------------
        do_prefit: bool
            Flag to enable prefit to background only (supported for unbinned fits only)
            The expected signal regions must be provided.

        **kwargs: dict
            Additional optional arguments:

            - prefit_excluded_regions: list
                List or list of lists with limits for the excluded regions

            - prefit_exclude_nsigma:
                Alternative argument to prefit_excluded_regions that excludes the regions around the signals,
                using the initialised mean and sigma parameters

        Returns
        -------------------------------------------------
        fit_result: zfit.minimizers.fitresult.FitResult
            The fit result
        """

        if self._data_handler_ is None:
            Logger('Data handler not specified', 'FATAL')

        self._raw_residuals_ = []
        self._raw_residual_variances_ = []
        self._std_residuals_ = []
        self._std_residual_variances_ = []

        self.__build_total_pdf()
        self.__build_total_pdf_binned()

        # do prefit
        if do_prefit:
            skip_prefit = False
            excluded_regions = []
            exclude_signals_nsigma = kwargs.get('prefit_exclude_nsigma', None)
            if exclude_signals_nsigma is not None:
                for ipdf, pdf_name in enumerate(self._name_signal_pdf_):
                    if pdf_name not in ['gaussian', 'cauchy', 'voigtian']:
                        Logger(f"Sigma parameter not defined for {pdf_name}, "
                               "cannot use nsigma for excluded regions in prefit", "Error")
                        skip_prefit = True
                    else:
                        mass_name = 'm' if pdf_name in ['cauchy', 'voigtian'] else 'mu'
                        mass = self._init_sgn_pars_[ipdf][mass_name]
                        sigma = self._init_sgn_pars_[ipdf]['sigma']
                        excluded_regions.append(
                            [mass - exclude_signals_nsigma * sigma, mass + exclude_signals_nsigma * sigma])
            else:
                excluded_regions = kwargs.get('prefit_excluded_regions', None)
                if excluded_regions is None:
                    Logger("To perform the prefit, you should set the excluded regions, skip", "Error")
                    skip_prefit = True

            if not skip_prefit:
                self.__prefit(excluded_regions)

        if self._data_handler_.get_is_binned():
            # chi2 loss
            if self._chi2_loss_:
                loss = zfit.loss.BinnedChi2(self._total_pdf_binned_, self._data_handler_.get_binned_data())
            # nll loss
            else:
                loss = zfit.loss.BinnedNLL(self._total_pdf_binned_, self._data_handler_.get_binned_data())
        else:
            loss = zfit.loss.UnbinnedNLL(model=self._total_pdf_, data=self._data_handler_.get_data())

        Logger("Performing FIT", "INFO")
        self._fit_result_ = self._minimizer_.minimize(loss=loss)
        Logger(self._fit_result_, 'RESULT')

        if self._fit_result_.hesse() == {}:
            if self._fit_result_.hesse(method='hesse_np') == {}:
                Logger('Impossible to compute hesse error', 'FATAL')

        self.__get_ratio_truncated()
        signal_fracs, _, _, signal_frac_errs, _, _ = self.__get_all_fracs()

        norm = self._total_pdf_norm_

        if len(self._fracs_) == 0:
            self._rawyield_[0] = self._data_handler_.get_norm()
            self._rawyield_err_[0] = np.sqrt(self._rawyield_[0])
        else:
            for i_pdf, _ in enumerate(self._signal_pdf_):
                self._rawyield_[i_pdf] = signal_fracs[i_pdf] * norm / self._ratio_truncated_
                self._rawyield_err_[i_pdf] = signal_frac_errs[i_pdf] * norm / self._ratio_truncated_

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

            - legend_loc: str
                location of the legend

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
        info_loc = kwargs.get('extra_info_loc', ['lower right', 'upper left'])
        legend_loc = kwargs.get('legend_loc', 'best')

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
        norm_total_pdf = self._total_pdf_norm_ * bin_sigma

        x_plot = np.linspace(limits[0], limits[1], num=num)
        total_func = zfit.run(self._total_pdf_.pdf(x_plot, norm_range=obs))
        signal_funcs, refl_funcs, bkg_funcs = ([] for _ in range(3))

        for signal_pdf in self._signal_pdf_:
            signal_funcs.append(zfit.run(signal_pdf.pdf(x_plot, norm_range=obs)))
        for refl_pdf in self._refl_pdf_:
            refl_funcs.append(zfit.run(refl_pdf.pdf(x_plot, norm_range=obs)))
        for bkg_pdf in self._background_pdf_:
            bkg_funcs.append(zfit.run(bkg_pdf.pdf(x_plot, norm_range=obs)))

        signal_fracs, bkg_fracs, refl_fracs, _, _, _ = self.__get_all_fracs()

        # first draw backgrounds
        for ibkg, (bkg_func, bkg_frac) in enumerate(zip(bkg_funcs, bkg_fracs)):
            plt.plot(x_plot, bkg_func * norm_total_pdf * bkg_frac / self._ratio_truncated_,
                     color=self._bkg_cmap_(ibkg), ls='--', label=self.label_bkg_pdf[ibkg])
        # then draw signals
        for isgn, (signal_func, frac) in enumerate(zip(signal_funcs, signal_fracs)):
            plt.plot(x_plot, signal_func * norm_total_pdf * frac / self._ratio_truncated_, color=self._sgn_cmap_(isgn))
            plt.fill_between(x_plot, signal_func * norm_total_pdf * frac / self._ratio_truncated_,
                             color=self._sgn_cmap_(isgn),
                             alpha=0.5, label=self.label_signal_pdf[isgn])

        # finally draw reflected signals (if any)
        for irefl, (refl_func, frac) in enumerate(zip(refl_funcs, refl_fracs)):
            if self._name_refl_pdf_[irefl] is None:
                continue
            plt.plot(x_plot, refl_func * norm_total_pdf * frac / self._ratio_truncated_, color=self._refl_cmap_(irefl))
            plt.fill_between(x_plot, refl_func * norm_total_pdf * frac / self._ratio_truncated_,
                             color=self._refl_cmap_(irefl),
                             alpha=0.5, label=f'reflected signal {irefl}')

        plt.plot(x_plot, total_func * norm_total_pdf, color='xkcd:blue', label='total fit')
        plt.xlim(limits[0], limits[1])
        plt.xlabel(axis_title)
        plt.ylabel(rf'counts / {(limits[1]-limits[0])/bins*1000:0.1f} MeV/$c^2$')
        if logy:
            plt.yscale('log')
            plt.ylim(min(total_func) * norm_total_pdf / 5, max(total_func) * norm_total_pdf * 5)
        else:
            plt.ylim(0., max(total_func) * norm_total_pdf * 1.5)

        if show_extra_info:
            # info on chi2/ndf
            chi2 = self.get_chi2()
            ndf = self.get_ndf()
            anchored_text_chi2 = AnchoredText(fr'$\chi^2 / \mathrm{{ndf}} =${chi2:.2f} / {ndf}',
                                              loc=info_loc[0],
                                              frameon=False)
            # signal and background info for all signals
            text = []
            for idx, _ in enumerate(self._name_signal_pdf_):
                mass, mass_unc = self.get_mass(idx)
                sigma, sigma_unc = None, None
                gamma, gamma_unc = None, None
                rawyield, rawyield_err = self.get_raw_yield(idx=idx)
                if self._name_signal_pdf_[idx] in [
                    'gausexptail', 'genergausexptailsymm', 'gaussian', 'crystalball',
                    'doublecb', 'doublecbsymm', 'voigtian', 'hist'
                    ]:
                    sigma, sigma_unc = self.get_sigma(idx)
                if self._name_signal_pdf_[idx] in ['cauchy', 'voigtian']:
                    gamma, gamma_unc = self.get_signal_parameter(idx, 'gamma')
                extra_info = fr'{self.label_signal_pdf[idx]}''\n' \
                    + fr'  $\mu = {mass*1000:.1f}\pm{mass_unc*1000:.1f}$ MeV$/c^2$''\n'
                if sigma is not None:
                    extra_info += fr'  $\sigma = {sigma*1000:.1f}\pm{sigma_unc*1000:.1f}$ MeV$/c^2$''\n'
                if gamma is not None:
                    extra_info += fr'  $\Gamma/2 = {gamma*1000:.1f}\pm{gamma_unc*1000:.1f}$ MeV$/c^2$''\n'

                extra_info += fr'  $S={rawyield:.0f} \pm {rawyield_err:.0f}$''\n'
                if self._name_background_pdf_[0] != 'nobkg':
                    if mass_range is not None:
                        interval = f'[{mass_range[0]:.3f}, {mass_range[1]:.3f}]'
                        bkg, bkg_err = self.get_background(idx=idx, min=mass_range[0], max=mass_range[1])
                        s_over_b, s_over_b_err = self.get_signal_over_background(idx=idx, min=mass_range[0],
                                                                                max=mass_range[1])
                        signif, signif_err = self.get_significance(idx=idx, min=mass_range[0], max=mass_range[1])
                        extra_info += fr'  $B({interval})={bkg:.0f} \pm {bkg_err:.0f}$''\n'
                        extra_info += fr'  $S/B({interval})={s_over_b:.2f} \pm {s_over_b_err:.2f}$''\n'
                        extra_info += fr'  Signif.$({interval})={signif:.1f} \pm {signif_err:.1f}$'
                    elif nhwhm is not None:
                        bkg, bkg_err = self.get_background(idx=idx, nhwhm=nhwhm)
                        s_over_b, s_over_b_err = self.get_signal_over_background(idx=idx, nhwhm=nhwhm)
                        signif, signif_err = self.get_significance(idx=idx, nhwhm=nhwhm)
                        extra_info += fr'  $B({nhwhm}~\mathrm{{HWHM}})=${bkg:.0f} $\pm$ {bkg_err:.0f}''\n'
                        extra_info += fr'  $S/B({nhwhm}~\mathrm{{HWHM}})=${s_over_b:.2f} $\pm$ {s_over_b_err:.2f}''\n'
                        extra_info += fr'  Signif.$({nhwhm}~\mathrm{{HWHM}})=${signif:.1f} $\pm$ {signif_err:.1f}'
                    else:
                        bkg, bkg_err = self.get_background(idx=idx, nsigma=nsigma)
                        s_over_b, s_over_b_err = self.get_signal_over_background(idx=idx, nsigma=nsigma)
                        signif, signif_err = self.get_significance(idx=idx, nsigma=nsigma)
                        extra_info += fr'  $B({nsigma}\sigma)=${bkg:.0f} $\pm$ {bkg_err:.0f}''\n'
                        extra_info += fr'  $S/B({nsigma}\sigma)=${s_over_b:.2f} $\pm$ {s_over_b_err:.2f}''\n'
                        extra_info += fr'  Signif.$({nsigma}\sigma)=${signif:.1f} $\pm$ {signif_err:.1f}'
                text.append(extra_info)
            concatenated_text = '\n'.join(text)
            anchored_text_signal = AnchoredText(concatenated_text, loc=info_loc[1], frameon=False)

            axs.add_artist(anchored_text_chi2)
            axs.add_artist(anchored_text_signal)

        plt.legend(loc=legend_loc)

        Logger('plot_mass_fit now returns a tuple (fig, axs) !', 'WARNING')

        return fig, axs

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
        norm = self._total_pdf_norm_ * bin_sigma

        x_plot = np.linspace(limits[0], limits[1], num=num)

        total_func = zfit.run(self._total_pdf_.pdf(x_plot, norm_range=obs))
        # write total_func
        self.__write_pdf(histname=f'total_func{suffix}', weight=total_func * norm, num=num,
                         filename=filename, option='update')

        signal_funcs, bkg_funcs, refl_funcs = ([] for _ in range(3))
        for signal_pdf in self._signal_pdf_:
            signal_funcs.append(zfit.run(signal_pdf.pdf(x_plot, norm_range=obs)))
        for refl_pdf in self._refl_pdf_:
            refl_funcs.append(zfit.run(refl_pdf.pdf(x_plot, norm_range=obs)))
        for bkg_pdf in self._background_pdf_:
            bkg_funcs.append(zfit.run(bkg_pdf.pdf(x_plot, norm_range=obs)))

        signal_fracs, bkg_fracs, refl_fracs, _, _, _ = self.__get_all_fracs()

        # first write backgrounds
        for ibkg, (bkg_func, bkg_frac) in enumerate(zip(bkg_funcs, bkg_fracs)):
            self.__write_pdf(histname=f'bkg_{ibkg}{suffix}',
                             weight=bkg_func * norm * bkg_frac / self._ratio_truncated_,
                             num=num, filename=filename, option='update')
        # then write signals
        for isgn, (frac, signal_func) in enumerate(zip(signal_funcs, signal_fracs)):
            self.__write_pdf(histname=f'signal_{isgn}{suffix}',
                             weight=signal_func * norm * frac / self._ratio_truncated_,
                             num=num, filename=filename, option='update')

        # finally write reflected signals
        for irefl, (frac, refl_func) in enumerate(zip(refl_funcs, refl_fracs)):
            if self._name_refl_pdf_[irefl] is None:
                continue
            self.__write_pdf(histname=f'refl_{irefl}{suffix}',
                             weight=refl_func * norm * frac / self._ratio_truncated_,
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
        nfreeparams = len(self._fit_result_.params)
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
            model_values = self._total_pdf_binned_.values() * norm
            # compute chi2
            for (data, model, data_variance) in zip(data_values, model_values, data_variances):
                chi2 += (data - model)**2/data_variance
            return chi2

        if self._limits_ != [self._data_handler_.get_limits()]:  # restricted fit limits
            Logger('chi2 not yet implemented with restricted fit limits', 'FATAL')
        # for unbinned data
        data_values = self._data_handler_.get_binned_data_from_unbinned_data()
        # access model predicted values
        model_values = self._total_pdf_binned_.values() * norm
        # compute chi2
        for (data, model) in zip(data_values, model_values):
            if data == 0:
                continue
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

        if len(self._raw_residuals_) == 0:
            self.__get_raw_residuals()

        # draw residuals
        plt.errorbar(
            self._data_handler_.get_bin_center(),
            self._raw_residuals_,
            xerr=None,
            yerr=np.sqrt(self._raw_residual_variances_),
            linestyle="None",
            elinewidth=1,
            capsize=0,
            color="black",
            marker="o",
            markersize=5,
            label="residuals"
        )
        bins = self._data_handler_.get_nbins()
        bin_sigma = (limits[1] - limits[0]) / bins
        norm = self._total_pdf_norm_ * bin_sigma

        x_plot = np.linspace(limits[0], limits[1], num=1000)
        signal_funcs, refl_funcs = ([] for _ in range(2))
        for signal_pdf in self._signal_pdf_:
            signal_funcs.append(zfit.run(signal_pdf.pdf(x_plot, norm_range=obs)))
        for refl_pdf in self._refl_pdf_:
            refl_funcs.append(zfit.run(refl_pdf.pdf(x_plot, norm_range=obs)))

        signal_fracs, _, refl_fracs, _, _, _ = self.__get_all_fracs()

        # draw signals
        for isgn, (signal_func, frac) in enumerate(zip(signal_funcs, signal_fracs)):
            plt.plot(x_plot, signal_func * norm * frac / self._ratio_truncated_, color=self._sgn_cmap_(isgn))
            plt.fill_between(x_plot, signal_func * norm * frac / self._ratio_truncated_, color=self._sgn_cmap_(isgn),
                             alpha=0.5, label=self.label_signal_pdf[isgn])

        # finally draw reflected signals (if any)
        is_there_refl = False
        for irefl, (refl_func, frac) in enumerate(zip(refl_funcs, refl_fracs)):
            if self._name_refl_pdf_[irefl] is None:
                continue
            is_there_refl = True
            plt.plot(x_plot, refl_func * norm * frac / self._ratio_truncated_, color=self._refl_cmap_(irefl))
            plt.fill_between(x_plot, refl_func * norm * frac / self._ratio_truncated_, color=self._refl_cmap_(irefl),
                             alpha=0.5, label=f'reflected signal {irefl}')

        # draw signal + reflected signals (if any)
        if is_there_refl:
            for isgn, (signal_func, refl_func, frac_sgn, frac_refl) in enumerate(
                    zip(signal_funcs, refl_funcs, signal_fracs, refl_fracs)):
                plt.plot(x_plot, (signal_func * frac_sgn + frac_refl * refl_func) * norm / self._ratio_truncated_,
                         color='xkcd:blue', label='total - bkg')

        plt.xlim(limits[0], limits[1])
        plt.xlabel(axis_title)
        plt.ylabel(rf'(data - fitted bkg) / {(limits[1]-limits[0])/bins*1000:0.1f} MeV/$c^2$')
        plt.legend(loc='best')

        return fig

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

        if len(self._std_residuals_) == 0:
            self.__get_std_residuals()
        # draw residuals
        plt.errorbar(bin_center,
                     self._std_residuals_,
                     xerr=None,
                     yerr=np.sqrt(self._std_residual_variances_),
                     linestyle="None",
                     elinewidth=1,
                     capsize=0,
                     color="black",
                     marker="o",
                     markersize=5,
                     label=None)

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

    def get_raw_yield_bincounting(self, idx=0, **kwargs):
        """
        Get the raw yield and its error via the bin-counting method

        Parameters
        -------------------------------------------------
        idx: int
            Index of the raw yield to be returned (default: 0)

        **kwargs: dict
            Additional optional arguments:

            - nsigma: float
                nsigma invariant-mass window around mean for signal counting

            - nhwhm: float
                number of hwhm invariant-mass window around mean for signal counting
                (alternative to nsigma)

            - min: float
                minimum value of invariant-mass for signal counting (alternative to nsigma)

            - max: float
                maximum value of invariant-mass for signal counting (alternative to nsigma)

        Returns
        -------------------------------------------------
        raw_yield: float
            The raw yield obtained from the bin counting
        raw_yield_err: float
            The raw yield error obtained from the bin counting
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
            if self._name_signal_pdf_[idx] not in [
                'gaussian', 'gausexptail', 'genergausexptailsymm', 'crystalball',
                'doublecb', 'doublecbsymm', 'voigtian', 'hist'
                ]:
                Logger('Sigma not defined, I cannot compute the signal for this pdf', 'ERROR')
                return 0., 0.
            mass, _ = self.get_mass(idx)
            sigma, _ = self.get_sigma(idx)
            min_value = mass - nsigma * sigma
            max_value = mass + nsigma * sigma

        if len(self._raw_residuals_) == 0:
            self.__get_raw_residuals()

        bin_centers = self._data_handler_.get_bin_center()
        bin_width = (bin_centers[1] - bin_centers[0]) / 2
        raw_yield, raw_yield_err = 0., 0.
        for residual, variance, bin_center in zip(self._raw_residuals_,
                                                  self._raw_residual_variances_, bin_centers):
            if bin_center - bin_width >= min_value and bin_center + bin_width <= max_value:
                raw_yield += residual
                raw_yield_err += variance
        raw_yield = float(raw_yield)
        raw_yield_err = np.sqrt(float(raw_yield_err))

        return raw_yield, raw_yield_err

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
        if 'hist' in self._name_signal_pdf_[idx]:
            hist = self._signal_pdf_[idx].to_hist()
            bin_limits = hist.to_numpy()[1]
            centres = [0.5 * (minn + maxx) for minn, maxx in zip(bin_limits[1:],  bin_limits[:-1])]
            counts = hist.values()
            mass = np.average(centres, weights=counts)
            mass_err = 0.
        else:
            mass_name = 'm' if self._name_signal_pdf_[idx] in ['cauchy', 'voigtian'] else 'mu'
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
        if self._name_signal_pdf_[idx] not in [
            'gaussian', 'gausexptail', 'genergausexptailsymm', 'crystalball',
            'doublecb', 'doublecbsymm', 'voigtian', 'hist'
            ]:
            Logger(f'Sigma parameter not defined for {self._name_signal_pdf_[idx]} pdf!', 'ERROR')
            return 0., 0.

        # if histogram, the rms is used as proxy
        if 'hist' in self._name_signal_pdf_[idx]:
            Logger(f'RMS used as proxy for sigma parameter of {self._name_signal_pdf_[idx]} pdf!', 'WARNING')
            mean = self.get_mass(idx)[0]
            hist = self._signal_pdf_[idx].to_hist()
            bin_limits = hist.to_numpy()[1]
            centres = [0.5 * (minn + maxx) for minn, maxx in zip(bin_limits[1:],  bin_limits[:-1])]
            counts = hist.values()
            sigma = np.sqrt(np.average((centres - mean)**2, weights=counts))
            sigma_err = 0.
        else:
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

    def get_signal_parameter(self, idx, par_name):
        """
        Get a signal parameter and its error

        Parameters
        -------------------------------------------------
        idx: int
            Index of the parameter to be returned (default: 0)
        par_name: str
            parameter to return

        Returns
        -------------------------------------------------
        parameter: float
            The parameter value obtained from the fit
        parameter_err: float
            The parameter error obtained from the fit

        """

        if par_name == 'gamma':
            Logger('The gamma parameter that you are getting is half of the width of a resonance,'
                   ' for more info check the Cauchy pdf defined here '
                   'https://zfit.readthedocs.io/en/latest/user_api/pdf/_generated/basic/zfit.pdf.Cauchy.html',
                   'WARNING')

        if self._fix_sgn_pars_[idx][par_name]:
            parameter = self._init_sgn_pars_[idx][par_name]
            parameter_err = 0.
        else:
            parameter = self._fit_result_.params[f'{self._name_}_{par_name}_signal{idx}']['value']
            parameter_err = self._fit_result_.params[f'{self._name_}_{par_name}_signal{idx}']['hesse']['error']

        return parameter, parameter_err

    def get_background_parameter(self, idx, par_name):
        """
        Get a background parameter and its error

        Parameters
        -------------------------------------------------
        idx: int
            Index of the parameter to be returned (default: 0)
        par_name: str
            parameter to return

        Returns
        -------------------------------------------------
        parameter: float
            The parameter value obtained from the fit
        parameter_err: float
            The parameter error obtained from the fit

        """

        if self._fix_bkg_pars_[idx][par_name]:
            parameter = self._init_bkg_pars_[idx][par_name]
            parameter_err = 0.
        else:
            parameter = self._fit_result_.params[f'{self._name_}_{par_name}_bkg{idx}']['value']
            parameter_err = self._fit_result_.params[f'{self._name_}_{par_name}_bkg{idx}']['hesse']['error']

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
            if self._name_signal_pdf_[idx] not in [
                'gaussian', 'gausexptail', 'genergausexptailsymm', 'crystalball',
                'doublecb', 'doublecbsymm', 'voigtian', 'hist'
                ]:
                Logger('Sigma not defined, I cannot compute the signal for this pdf', 'ERROR')
                return 0., 0.
            mass, _ = self.get_mass(idx)
            sigma, _ = self.get_sigma(idx)
            min_value = mass - nsigma * sigma
            max_value = mass + nsigma * sigma

        # pylint: disable=missing-kwoa
        signal = self._signal_pdf_[idx].integrate((min_value, max_value))

        signal_fracs, _, refl_fracs, signal_err_fracs, _, _ = self.__get_all_fracs()

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
                frac = 1. - sum(signal_fracs) - sum(refl_fracs)
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
            if self._name_signal_pdf_[idx] not in [
                'gaussian', 'gausexptail', 'genergausexptailsymm', 'crystalball',
                'doublecb', 'doublecbsymm', 'voigtian', 'hist'
                ]:
                Logger('Sigma not defined, I cannot compute the signal for this pdf', 'ERROR')
                return 0., 0.
            mass, _ = self.get_mass(idx)
            sigma, _ = self.get_sigma(idx)
            min_value = mass - nsigma * sigma
            max_value = mass + nsigma * sigma

        _, bkg_fracs, _, _, bkg_err_fracs, _ = self.__get_all_fracs()

        limits = self._data_handler_.get_limits()
        min_value = max(min_value, limits[0])
        max_value = min(max_value, limits[1])

        # pylint: disable=missing-kwoa
        background, background_err = 0., 0.
        for idx2, bkg in enumerate(self._background_pdf_):

            norm = self._total_pdf_norm_ * bkg_fracs[idx2]
            norm_err = norm * bkg_err_fracs[idx2]

            bkg_int = float(bkg.integrate((min_value, max_value)))
            background += bkg_int * norm
            background_err += (bkg_int * norm_err)**2

        background /= self._ratio_truncated_
        background_err = np.sqrt(background_err) / self._ratio_truncated_

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

    def get_sweights(self):
        """
        Calculate sWeights for signal and background components.

        Returns:
            dict: A dictionary containing sWeights for 'signal' and 'bkg' components.
        """

        signal_pdf_extended = []
        refl_pdf_extended = []
        bkg_pdf_extended = []

        norm = self._data_handler_.get_norm()

        signal_fracs, bkg_fracs, refl_fracs, _, _, _ = self.__get_all_fracs()
        bkg_fracs.append(1-sum(bkg_fracs)-sum(signal_fracs)-sum(refl_fracs))

        total_signal_yields = 0.
        total_bkg_yields = 0.
        for pdf, frac in zip(self._signal_pdf_, signal_fracs):
            signal_pdf_extended.append(pdf.create_extended(frac * norm))
            total_signal_yields += frac * norm
        for pdf, frac in zip(self._refl_pdf_, refl_fracs):
            refl_pdf_extended.append(pdf.create_extended(frac * norm))
            total_signal_yields += frac * norm
        for pdf, frac in zip(self._background_pdf_, bkg_fracs):
            bkg_pdf_extended.append(pdf.create_extended(frac * norm))
            total_bkg_yields += frac * norm

        total_signal_yields_par = zfit.Parameter('signal_yield', total_signal_yields)
        total_bkg_yields_par = zfit.Parameter('bkg_yield', total_bkg_yields)

        if len(signal_pdf_extended + refl_pdf_extended) > 1:
            signal_pdf_for_sweights = zfit.pdf.SumPDF(
                signal_pdf_extended + refl_pdf_extended,
                extended=total_signal_yields_par
            )
        else:
            signal_pdf_for_sweights = signal_pdf_extended[0]

        if len(bkg_pdf_extended) > 1:
            bkg_pdf_for_sweights = zfit.pdf.SumPDF(
                bkg_pdf_extended,
                extended=total_bkg_yields_par
            )
        else:
            bkg_pdf_for_sweights = bkg_pdf_extended[0]

        total_pdf_for_sweights = zfit.pdf.SumPDF([signal_pdf_for_sweights, bkg_pdf_for_sweights])
        sweights = compute_sweights(total_pdf_for_sweights, self._data_handler_.get_data())
        sweights_labels = list(sweights.keys())

        return {'signal': sweights[sweights_labels[0]], 'bkg': sweights[sweights_labels[1]]}

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
        mass_name = 'm' if self._name_signal_pdf_[idx] in ['cauchy', 'voigtian'] else 'mu'
        mass = 0.
        if 'mass' in kwargs:
            mass = kwargs['mass']
        elif 'pdg_id' in kwargs:
            mass = self.pdg_api.get_particle_by_mcid(kwargs['pdg_id']).mass
        elif 'pdg_name' in kwargs:
            mass = self.pdg_api.get_particle_by_name(kwargs['pdg_name']).mass
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
        if par_name == 'gamma':
            Logger('The gamma parameter that you are setting is half of the width of a resonance,'
                   ' for more info check the Cauchy pdf defined here '
                   'https://zfit.readthedocs.io/en/latest/user_api/pdf/_generated/basic/zfit.pdf.Cauchy.html',
                   'WARNING')

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

    def __get_parameter_fix_frac(self, factor, name):
        """
        Get the parameter fix for the fraction

        Parameters
        -------------------------------------------------
        idx_pdf: int
            Index of the pdf
        target_pdf: int
            Index of the target pdf
        factor: float
            Factor to multiply the fraction parameter
        """
        return zfit.Parameter(name, factor, floating=False)

    def __check_consistency_fix_frac(self, idx_pdf, target_pdf, idx_pdf_type, target_pdf_type):
        """
        Checks the consistency of fixing the fraction between PDFs.

        Parameters
        -------------------------------------------------
        idx_pdf: int
            The index of the PDF to check.
        target_pdf: int
            The target PDF to compare against.
        idx_pdf_type: 'signal' or 'bkg'
            The type of the PDF to check.
        target_pdf_type: 'signal' or 'bkg'
            The type of the target PDF to compare against.
        """
        if idx_pdf_type == target_pdf_type and idx_pdf == target_pdf:
            Logger(
                f'Index {idx_pdf} is the same as {target_pdf},'
                'cannot constrain the fraction to itself',
                'FATAL'
            )
        if target_pdf_type == 'signal' and target_pdf > len(self._signal_pdf_):
            Logger(
                f'Target signal index {target_pdf} is out of range',
                'FATAL'
            )
        if target_pdf_type == 'bkg' and target_pdf > len(self._background_pdf_):
            Logger(
                f'Target background index {target_pdf} is out of range',
                'FATAL'
            )

    def fix_signal_frac_to_signal_pdf(self, idx_pdf, target_pdf, factor=1):
        """
        Fix the frac parameter of a signal to the frac parameter of another signal

        Parameters
        -------------------------------------------------
        idx_pdf: int
            Index of the signal fraction to be fixed
        target_pdf: int
            Index of the signal fraction to be used as reference
        factor: float
            Factor to multiply the frac parameter of the target signal
        """
        self.__check_consistency_fix_frac(idx_pdf, target_pdf, 'signal', 'signal')
        factor_par = self.__get_parameter_fix_frac(
            factor, f'factor_signal{idx_pdf}_constrained_to_signal{target_pdf}'
        )
        self._fix_fracs_to_pdfs_.append({
            'fixed_pdf_idx': idx_pdf,
            'target_pdf_idx': target_pdf,
            'factor': factor_par,
            'fixed_pdf_type': 'signal',
            'target_pdf_type': 'signal'
        })

    def fix_signal_frac_to_bkg_pdf(self, idx_pdf, target_pdf, factor=1):
        """
        Fix the frac parameter of a signal to the frac parameter of a background

        Parameters
        -------------------------------------------------
        idx_pdf: int
            Index of the signal fraction to be fixed
        target_pdf: int
            Index of the background fraction to be used as reference
        factor: float
            Factor to multiply the frac parameter of the target background
        """
        self.__check_consistency_fix_frac(idx_pdf, target_pdf, 'signal', 'bkg')
        factor_par = self.__get_parameter_fix_frac(
            factor, f'factor_signal{idx_pdf}_constrained_to_bkg{target_pdf}'
        )
        self._fix_fracs_to_pdfs_.append({
            'fixed_pdf_idx': idx_pdf,
            'target_pdf_idx': target_pdf,
            'factor': factor_par,
            'fixed_pdf_type': 'signal',
            'target_pdf_type': 'bkg'
        })

    def fix_bkg_frac_to_signal_pdf(self, idx_pdf, target_pdf, factor=1):
        """
        Fix the frac parameter of a background to the frac parameter of a signal

        Parameters
        -------------------------------------------------
        idx_pdf: int
            Index of the background fraction to be fixed
        target_pdf: int
            Index of the signal fraction to be used as reference
        factor: float
            Factor to multiply the frac parameter of the target signal
        """
        self.__check_consistency_fix_frac(idx_pdf, target_pdf, 'bkg', 'signal')
        factor_par = self.__get_parameter_fix_frac(
            factor, f'factor_bkg{idx_pdf}_constrained_to_signal{target_pdf}'
        )
        self._fix_fracs_to_pdfs_.append({
            'fixed_pdf_idx': idx_pdf,
            'target_pdf_idx': target_pdf,
            'factor': factor_par,
            'fixed_pdf_type': 'bkg',
            'target_pdf_type': 'signal'
        })

    def fix_bkg_frac_to_bkg_pdf(self, idx_pdf, target_pdf, factor=1):
        """
        Fix the frac parameter of a background to the frac parameter of another background

        Parameters
        -------------------------------------------------
        idx_pdf: int
            Index of the background fraction to be fixed
        target_pdf: int
            Index of the background fraction to be used as reference
        factor: float
            Factor to multiply the frac parameter of the target background
        """
        self.__check_consistency_fix_frac(idx_pdf, target_pdf, 'bkg', 'bkg')
        factor_par = self.__get_parameter_fix_frac(
            factor, f'factor_bkg{idx_pdf}_constrained_to_bkg{target_pdf}'
        )
        self._fix_fracs_to_pdfs_.append({
            'fixed_pdf_idx': idx_pdf,
            'target_pdf_idx': target_pdf,
            'factor': factor_par,
            'fixed_pdf_type': 'bkg',
            'target_pdf_type': 'bkg'
        })

    # pylint: disable=line-too-long
    def set_signal_template(self, idx, sample):
        """
        Set sample and options for signal template

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal
        sample: flarefly.DataHandler
            Data sample for histogram template
        """

        self._hist_signal_sample_[idx] = sample

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
    def set_reflection_template(self, idx, sample, r_over_s):
        """
        Set sample and options for reflected signal template

        Parameters
        -------------------------------------------------
        idx: int
            Index of the reflected signal
        sample: flarefly.DataHandler
            Data sample for histogram template
        r_over_s: float
            R/S ratio
        """

        self._hist_refl_sample_[idx] = sample
        self._refl_over_sgn_[idx] = r_over_s

    # pylint: disable=line-too-long
    def set_reflection_kde(self, idx, sample, r_over_s, **kwargs):
        """
        Set sample and options for reflected signal kde

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal
        sample: flarefly.DataHandler
            Data sample for Kernel Density Estimation
        r_over_s: float
            R/S ratio
        **kwargs: dict
            Arguments for kde options. See
            https://zfit.readthedocs.io/en/latest/user_api/pdf/_generated/kde_api/zfit.pdf.KDE1DimGrid.html#zfit.pdf.KDE1DimGrid
            for more details
        """

        self._kde_signal_sample_[idx] = sample
        self._kde_signal_option_[idx] = kwargs
        self._refl_over_sgn_[idx] = r_over_s

    # pylint: disable=line-too-long
    def set_background_template(self, idx, sample):
        """
        Set sample and options for background template histogram

        Parameters
        -------------------------------------------------
        idx: int
            Index of the background
        sample: flarefly.DataHandler
            Data sample for template histogram
        """

        limits_bkg = sample.get_limits()
        limits_data = self._data_handler_.get_limits()
        if not np.isclose(limits_bkg[0], limits_data[0], rtol=1e-05, atol=1e-08, equal_nan=False) or \
                not np.isclose(limits_bkg[1], limits_data[1], rtol=1e-05, atol=1e-08, equal_nan=False):
            Logger(f'The data and the background template {idx} have different limits:'
                   f' \n       -> background template: {limits_bkg}, data -> {limits_data}', 'FATAL')

        self._hist_bkg_sample_[idx] = sample

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

    def get_name_signal_pdf(self):
        """
        Return the signal pdf names
        """
        return self._name_signal_pdf_.copy()

    def get_name_background_pdf(self):
        """
        Return the background pdf names
        """
        return self._name_background_pdf_.copy()

    def get_name_refl_pdf(self):
        """
        Return the reflection pdf names
        """
        return self._name_refl_pdf_.copy()

    def get_signal_pars(self):
        """
        Return the signal parameters
        """
        fracs = self.__get_all_fracs()
        signal_pars = []
        for i_sgn, sgn_par in enumerate(self._sgn_pars_):
            signal_pars.append({})
            for key, value in sgn_par.items():
                par_name = key.split(f'{self._name_}_')[-1]
                par_name = par_name.split(f'_signal{i_sgn}')[0]
                signal_pars[-1][par_name] = value.numpy()
            signal_pars[-1]['frac'] = fracs[0][i_sgn]
        return signal_pars

    def get_signal_pars_uncs(self):
        """
        Return the signal parameters uncertainties
        """
        fracs = self.__get_all_fracs()
        signal_pars_uncs = []
        for i_sgn, sgn_par in enumerate(self._sgn_pars_):
            signal_pars_uncs.append({})
            for key in sgn_par:
                par_name = key.split(f'{self._name_}_')[-1]
                par_name = par_name.split(f'_signal{i_sgn}')[0]
                try:
                    par_unc = self._fit_result_.params[key]['hesse']['error']
                except KeyError:
                    par_unc = 0.
                signal_pars_uncs[-1][par_name] = par_unc
            signal_pars_uncs[-1]['frac'] = fracs[3][i_sgn]
        return signal_pars_uncs

    def get_bkg_pars(self):
        """
        Return the background parameters
        """
        fracs = self.__get_all_fracs()
        bkg_pars = []
        for i_bkg, bkg_par in enumerate(self._bkg_pars_):
            bkg_pars.append({})
            for key, value in bkg_par.items():
                par_name = key.split(f'{self._name_}_')[-1]
                par_name = par_name.split(f'_bkg{i_bkg}')[0]
                bkg_pars[-1][par_name] = value.numpy()
            bkg_pars[-1]['frac'] = fracs[1][i_bkg]
        return bkg_pars

    def get_bkg_pars_uncs(self):
        """
        Return the background parameters uncertainties
        """
        fracs = self.__get_all_fracs()
        bkg_pars_uncs = []
        for i_bkg, bkg_par in enumerate(self._bkg_pars_):
            bkg_pars_uncs.append({})
            for key in bkg_par:
                par_name = key.split(f'{self._name_}_')[-1]
                par_name = par_name.split(f'_bkg{i_bkg}')[0]
                try:
                    par_unc = self._fit_result_.params[key]['hesse']['error']
                except KeyError:
                    par_unc = 0.
                bkg_pars_uncs[-1][par_name] = par_unc
            bkg_pars_uncs[-1]['frac'] = fracs[4][i_bkg]
        return bkg_pars_uncs
