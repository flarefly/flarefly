"""
Module containing the class used to perform mass fits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import zfit
import mplhep
from flarefly.utils import Logger


# pylint: disable=too-many-instance-attributes
class F2MassFitter:
    """
    Class used to perform mass fits with the zfit library
    https://github.com/zfit/zfit
    """

    def __init__(self, data_handler, name_signal_pdf, name_background_pdf, name=""):
        """
        Initialize the F2MassFitter class
        Parameters
        -------------------------------------------------
        data_handler: flarefly.DataHandler
            The data handler containing the data to fit
        name_signal_pdf: list
            The list of names for the signal pdfs. The possible options are:
                - 'gaussian'
                - 'crystalball'
                - 'cauchy'
                - 'kde_exact' (requires to set the datasample and options)
                - 'kde_grid' (requires to set the datasample and options)
                - 'kde_fft' (requires to set the datasample and options)
                - 'kde_isj' (requires to set the datasample and options)
        name_background_pdf: list
            The list of names of the background pdfs. The possible options are:
                - 'nobkg'
                - 'expo'
                - 'kde_exact' (requires to set the datasample and options)
                - 'kde_grid' (requires to set the datasample and options)
                - 'kde_fft' (requires to set the datasample and options)
                - 'kde_isj' (requires to set the datasample and options)
        name: str
            Optional name for the fitter,
            needed in case of multiple fitters defined in the same script
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
        # pylint: disable=fixme
        self.__prefit() #TODO: implement me

        if self._data_handler_.get_is_binned():
            self._total_pdf_binned_ = zfit.pdf.BinnedFromUnbinnedPDF(self._total_pdf_,
                                                                     self._data_handler_.get_obs())
            nll = zfit.loss.BinnedNLL(self._total_pdf_binned_,
                                      self._data_handler_.get_binned_data())
        else:
            nll = zfit.loss.UnbinnedNLL(
                model=self._total_pdf_, data=self._data_handler_.get_data())

        self._fit_result_ = self._minimizer_.minimize(loss=nll)
        self._fit_result_.hesse()
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

    # pylint: disable=too-many-statements
    def plot_mass_fit(self, **kwargs):
        """
        Plot the mass fit

        Parameters
        -------------------------------------------------
        kwargs:
            - style: str
                style to be used (see https://github.com/scikit-hep/mplhep for more details)
            - logy: bool
                log scale in y axis
            - figsize: tuple
                size of the figure
            - bins: int
                number of bins in case of unbinned fit
            - axis_title: str
                x-axis title

        Returns
        -------------------------------------------------
        fig: matplotlib.figure.Figure
            figure containing the mass fit plot
        """

        style = kwargs.get('style', 'LHCb2')
        logy = kwargs.get('logy', False)
        figsize = kwargs.get('figsize', (7, 7))
        bins = kwargs.get('bins', 100)
        axis_title = kwargs.get('axis_title', self._data_handler_.get_var_name())

        mplhep.style.use(style)

        obs = self._data_handler_.get_obs()
        limits = self._data_handler_.get_limits()

        fig = plt.figure(figsize=figsize)
        if self._data_handler_.get_is_binned():
            hist = self._data_handler_.get_binned_data().to_hist()
            hist.plot(yerr=True, color='black', histtype='errorbar',
                      label='data')
            bins = len(self._data_handler_.get_binned_data().values())
            bin_sigma = (limits[1] - limits[0]) / bins
            norm = self._data_handler_.get_norm() * bin_sigma
        else:
            data_np = zfit.run(self._data_handler_.get_data()[:, 0])
            counts, bin_edges = np.histogram(data_np, bins, range=(limits[0], limits[1]))
            mplhep.histplot((counts, bin_edges), yerr=True, color='black', histtype='errorbar',
                            label='data')
            norm = data_np.shape[0] / bins * obs.area()

        x_plot = np.linspace(limits[0], limits[1], num=1000)
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
        base_sgn_cmap = plt.cm.get_cmap('viridis', len(signal_funcs) * 4)
        sgn_cmap = ListedColormap(base_sgn_cmap(np.linspace(0.4, 0.65, len(signal_funcs))))
        for isgn, (frac, signal_func) in enumerate(zip(signal_funcs, signal_fracs)):
            plt.plot(x_plot, signal_func * norm * frac, color=sgn_cmap(isgn))
            plt.fill_between(x_plot, signal_func * norm * frac, color=sgn_cmap(isgn),
                             alpha=0.5, label=f'signal {isgn}')

        plt.plot(x_plot, total_func * norm, color='xkcd:blue', label='total fit')
        plt.xlim(limits[0], limits[1])
        plt.xlabel(axis_title)
        plt.ylabel(rf'counts / {(limits[1]-limits[0])/bins*1000:0.1f} MeV/$c^2$')
        plt.legend(loc='best')
        if logy:
            plt.yscale('log')
            plt.ylim(min(total_func) * norm / 5, max(total_func) * norm * 5)

        return fig

    def get_fit_result(self):
        """
        Get the fit result

        Returns
        -------------------------------------------------
        fit_result: zfit.minimizers.fitresult.FitResult
            The fit result
        """
        return self._fit_result_

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
        if self._name_signal_pdf_[idx] not in ['gaussian', 'crystalball']:
            Logger(f'Sigma parameter not defined for {self._name_signal_pdf_[idx]} pdf!', 'ERROR')
            return 0., 0.

        if self._fix_sgn_pars_[idx]['sigma']:
            sigma = self._init_sgn_pars_[idx]['sigma']
            sigma_err = 0.
        else:
            sigma = self._fit_result_.params[f'{self._name_}_sigma_signal{idx}']['value']
            sigma_err = self._fit_result_.params[f'{self._name_}_sigma_signal{idx}']['hesse']['error']

        return sigma, sigma_err

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

    def get_signal(self, idx=0, nsigma=3):
        """
        Get the signal and its error in mass +- nsigma * width

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal to be returned
        nsigma: float
            nsigma window for signal computation

        Returns
        -------------------------------------------------
        signal: float
            The signal value obtained from the fit
        signal_err: float
            The signal error obtained from the fit
        """

        if self._name_signal_pdf_[idx] not in ['gaussian', 'crystalball']:
            # pylint: disable=fixme
            # TODO: add possibility to compute signal not based on nsigma
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

    def get_background(self, idx=0, nsigma=3):
        """
        Get the background and its error in mass +- nsigma * width

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal to be used to compute nsigma window
        nsigma: float
            nsigma window for background computation

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

        return background, background_err

    def get_signal_over_background(self, idx=0, nsigma=3):
        """
        Get the S/B ratio and its error in mass +- nsigma * width

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal to be used to compute nsigma window
        nsigma: float
            nsigma window for background computation

        Returns
        -------------------------------------------------
        signal_over_background: float
            The S/B value obtained from the fit
        signal_over_background_err: float
            The S/B error obtained from the fit
        """

        signal = self.get_signal(idx, nsigma)
        bkg = self.get_background(idx, nsigma)
        signal_over_background = signal[0]/bkg[0]
        signal_over_background_err = np.sqrt(signal[1]**2/signal[0]**2 + bkg[1]**2/bkg[0]**2)
        signal_over_background_err *= signal_over_background

        return signal_over_background, signal_over_background_err

    def get_significance(self, idx=0, nsigma=3):
        """
        Get the significance and its error in mass +- nsigma * width

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal to be used to compute nsigma window
        nsigma: float
            nsigma window for background computation

        Returns
        -------------------------------------------------
        significance: float
            The significance value obtained from the fit
        significance_err: float
            The significance error obtained from the fit
        """

        signal = self.get_signal(idx, nsigma)
        bkg = self.get_background(idx, nsigma)
        significance = signal[0]/np.sqrt(signal[0]+bkg[0])
        sig_plus_bkg = signal[0] + bkg[0]

        significance_err = significance*np.sqrt(
            (signal[1]**2 + bkg[1]**2) / (4. * sig_plus_bkg**2) + (
                bkg[0]/sig_plus_bkg) * signal[1]**2 / signal[0]**2)

        return significance, significance_err

    def set_particle_mass(self, idx, mass, **kwargs):
        """
        Set the particle mass

        Parameters
        -------------------------------------------------
        idx: int
            Index of the signal
        mass: float
            The mass to be set
        **kwargs: dict
            Additional optional arguments:
            - limits: list
                minimum and maximum limits for the mass parameter
            - fix: bool
                fix the mass parameter
        """
        mass_name = 'm' if self._name_signal_pdf_[idx] == 'cauchy' else 'mu'
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
            https://zfit.readthedocs.io/en/latest/user_api/pdf/
            _generated/kde_api/zfit.pdf.KDE1DimGrid.html#zfit.pdf.KDE1DimGrid
            for more details
        """

        self._kde_signal_sample_[idx] = sample
        self._kde_signal_option_[idx] = kwargs

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
            https://zfit.readthedocs.io/en/latest/user_api/pdf/
            _generated/kde_api/zfit.pdf.KDE1DimGrid.html#zfit.pdf.KDE1DimGrid
            for more details
        """

        self._kde_bkg_sample_[idx] = sample
        self._kde_bkg_option_[idx] = kwargs
