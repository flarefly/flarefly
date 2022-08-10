"""
Module containing the class used to perform mass fits
"""

import numpy as np
import matplotlib.pyplot as plt

import zfit
import mplhep
from flarefly.utils import Logger


# pylint: disable=too-many-instance-attributes
class F2MassFitter:
    """
    Class used to perform mass fits with the zfit library
    https://github.com/zfit/zfit
    """

    def __init__(self, data_handler=None, name_signal_pdf=None, name_background_pdf=None):
        """
        Initialize the F2MassFitter class
        Parameters
        -------------------------------------------------
        data_handler: flarefly.DataHandler
            The data handler containing the data to fit
        name_signal_pdf: str
            The name of the signal pdf. The possible options are:
            - 'gaussian'
            - 'crystalball'
        name_background_pdf: str
            The name of the background pdf. The possible options are:
            - 'expo'
        """
        self._data_handler_ = data_handler
        self._name_signal_pdf_ = name_signal_pdf
        self._name_background_pdf_ = name_background_pdf
        self._signal_pdf_ = None
        self._background_pdf_ = None
        self._total_pdf_ = None
        self._total_pdf_binned_ = None
        self._fit_result_ = None
        self._init_mass_ = 1.865
        self._init_width_ = 0.01
        self._mass_ = None
        self._width_ = None
        self._signal_frac_ = None
        self._alpha_ = None
        self._nsig_ = None
        self._bkg_pars_ = [None]
        self._minimizer_ = zfit.minimize.Minuit(verbosity=7)
        self._rawyield_ = 0.
        self._rawyield_err_ = 0.
        self._name_secpeak_pdf_ = 'nosecpeak'
        self._init_mass_secpeak_ = 1.870
        self._init_width_secpeak_ = 0.01
        self._mass_secpeak_ = None
        self._width_secpeak_ = None
        self._alpha_secpeak_ = None
        self._nsig_secpeak_ = None
        self._secpeak_frac_ = None
        self._secpeak_pdf_ = None

        zfit.settings.advanced_warnings.all = False
        zfit.settings.changed_warnings.all = False

    # pylint: disable=too-many-branches
    def __build_signal_pdf(self, obs):
        """
        Helper function to compose the signal pdf
        """

        if self._name_signal_pdf_ == 'gaussian':
            if self._mass_ is None:
                self._mass_ = zfit.Parameter('mass', self._init_mass_)
            else:
                self._mass_.set_value(self._init_mass_)
            if self._width_ is None:
                self._width_ = zfit.Parameter('width', self._init_width_)
            else:
                self._width_.set_value(self._init_width_)
            self._signal_pdf_ = zfit.pdf.Gauss(obs=obs, mu=self._mass_, sigma=self._width_)
        elif self._name_signal_pdf_ == 'crystalball':
            if self._mass_ is None:
                self._mass_ = zfit.Parameter('mass', self._init_mass_)
            else:
                self._mass_.set_value(self._init_mass_)
            if self._width_ is None:
                self._width_ = zfit.Parameter('width', self._init_width_)
            else:
                self._width_.set_value(self._init_width_)
            if self._alpha_ is None:
                self._alpha_ = zfit.Parameter('alpha', 0.5)
            if self._nsig_ is None:
                self._nsig_ = zfit.Parameter('nsig', 1.)
            self._signal_pdf_ = zfit.pdf.CrystalBall(obs=obs, mu=self._mass_, sigma=self._width_,
                                                     alpha=self._alpha_, n=self._nsig_)
        else:
            Logger('Signal pdf not supported', 'FATAL')

    def __build_background_pdf(self, obs):
        """
        Helper function to compose the background pdf
        """

        if self._name_background_pdf_ == 'nobkg':
            Logger('Performing fit with no bkg pdf', 'WARNING')
        elif self._name_background_pdf_ == 'expo':
            if self._bkg_pars_[0] is None:
                self._bkg_pars_[0] = zfit.Parameter('bkg_p0', 0.1)
            self._background_pdf_ = zfit.pdf.Exponential(self._bkg_pars_[0], obs=obs)
        else:
            Logger('Background pdf not supported', 'FATAL')

    # pylint: disable=too-many-branches
    def __build_secpeak_pdf(self, obs):
        """
        Helper function to compose the second peak pdf
        """

        if self._name_secpeak_pdf_ == 'noseckpeak':
            return
        elif self._name_secpeak_pdf_ == 'gaussian':
            if self._mass_secpeak_ is None:
                self._mass_secpeak_ = zfit.Parameter('mass_secpeak', self._init_mass_secpeak_)
            else:
                self._mass_secpeak_.set_value(self._init_mass_secpeak_)
            if self._width_secpeak_ is None:
                self._width_secpeak_ = zfit.Parameter('width_secpeak', self._init_width_secpeak_)
            else:
                self._width_secpeak_.set_value(self._init_width_secpeak_)
            self._secpeak_pdf_ = zfit.pdf.Gauss(obs=obs, mu=self._mass_secpeak_,
                                                sigma=self._width_secpeak_)
        elif self._name_secpeak_pdf_ == 'crystalball':
            if self._mass_secpeak_ is None:
                self._mass_secpeak_ = zfit.Parameter('mass_secpeak', self._init_mass_secpeak_)
            else:
                self._mass_secpeak_.set_value(self._init_mass_secpeak_)
            if self._width_secpeak_ is None:
                self._width_secpeak_ = zfit.Parameter('width_secpeak', self._init_width_secpeak_)
            else:
                self._width_secpeak_.set_value(self._init_width_secpeak_)
            if self._alpha_secpeak_ is None:
                self._alpha_secpeak_ = zfit.Parameter('alpha_secpeak', 0.5)
            if self._nsig_secpeak_ is None:
                self._nsig_secpeak_ = zfit.Parameter('nsig_secpeak', 1.)
            self._secpeak_pdf_ = zfit.pdf.CrystalBall(obs=obs, mu=self._mass_secpeak_,
                                                      sigma=self._width_secpeak_,
                                                      alpha=self._alpha_secpeak_,
                                                      n=self._nsig_secpeak_)
        else:
            Logger('Second peak pdf not supported, the second peak will not be included.', 'ERORR')

    def __build_total_pdf(self):
        """
        Helper function to compose the total pdf
        """

        obs = self._data_handler_.get_obs()

        # order of the pdfs is signal, second peak, background

        self.__build_signal_pdf(obs)
        self.__build_secpeak_pdf(obs)
        self.__build_background_pdf(obs)

        list_pdfs = [self._signal_pdf_]
        if self._secpeak_pdf_:
            list_pdfs.append(self._secpeak_pdf_)
            if not self._secpeak_frac_:
                self._secpeak_frac_ = zfit.Parameter('secpeak_frac', 0.1, 0., 1.)
        if self._background_pdf_:
            list_pdfs.append(self._background_pdf_)

        n_pdfs = len(list_pdfs)
        list_fracs = []
        if n_pdfs > 1:
            if not self._signal_frac_:
                self._signal_frac_ = zfit.Parameter('sig_frac', 0.1, 0., 1.)
            list_fracs.append(self._signal_frac_)
            if n_pdfs > 2:
                list_fracs.append(self._secpeak_frac_)

            self._total_pdf_ = zfit.pdf.SumPDF(list_pdfs, list_fracs)
        else:
            self._total_pdf_ = self._signal_pdf_

    def __prefit(self):
        """
        Helper function to perform a prefit to the sidebands
        """

        Logger('Prefit step to be implemented', 'WARNING')

    def set_particle_mass(self, mass):
        """
        Set the particle mass

        Parameters
        -------------------------------------------------
        mass: float
            The mass to be set
        """
        self._init_mass_ = mass

    def set_peak_width(self, width):
        """
        Set the expected particle peak width

        Parameters
        -------------------------------------------------
        width: float
            The width to be set
        """
        self._init_width_ = width

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
        self.__prefit()

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

        if self._background_pdf_:
            norm = self._data_handler_.get_norm()
            self._rawyield_ = self._fit_result_.params['sig_frac']['value'] * norm
            self._rawyield_err_ = self._fit_result_.params['sig_frac']['hesse']['error'] * norm
        else:
            self._rawyield_ = self._data_handler_.get_norm()
            self._rawyield_err_ = np.sqrt(self._rawyield_)

        return self._fit_result_

    def plot_mass_fit(self, style='LHCb2', logy=False):
        """
        Plot the mass fit

        Parameters
        -------------------------------------------------
        style: str
            style to be used (see https://github.com/scikit-hep/mplhep for more details)
        logy: bool
            log scale in y axis

        Returns
        -------------------------------------------------
        fig: matplotlib.figure.Figure
            figure containing the mass fit plot
        """
        mplhep.style.use(style)

        obs = self._data_handler_.get_obs()
        limits = self._data_handler_.get_limits()

        fig = plt.figure(figsize=(7, 7))
        if self._data_handler_.get_is_binned():
            hist = self._data_handler_.get_binned_data().to_hist()
            hist.plot(yerr=True, color='black', histtype='errorbar',
                      label='data')
            bins = len(self._data_handler_.get_binned_data().values())
            bin_width = (limits[1] - limits[0]) / bins
            norm = self._data_handler_.get_norm() * bin_width
        else:
            data_np = zfit.run(self._data_handler_.get_data()[:, 0])
            bins = 100
            counts, bin_edges = np.histogram(data_np, bins, range=(limits[0], limits[1]))
            mplhep.histplot((counts, bin_edges), yerr=True, color='black', histtype='errorbar',
                            label='data')
            norm = data_np.shape[0] / bins * obs.area()

        x_plot = np.linspace(limits[0], limits[1], num=1000)
        total_func = zfit.run(self._total_pdf_.pdf(x_plot, norm_range=obs))
        signal_func = zfit.run(self._signal_pdf_.pdf(x_plot, norm_range=obs))

        signal_frac = 1.
        secpeak_frac = 0.
        if self._signal_frac_:
            signal_frac = self._fit_result_.params['sig_frac']['value']
        if self._secpeak_frac_:
            secpeak_frac = self._fit_result_.params['secpeak_frac']['value']

        if self._name_background_pdf_ != "nobkg":
            bkg_func = zfit.run(self._background_pdf_.pdf(x_plot, norm_range=obs))
            plt.plot(x_plot, bkg_func * norm * (1.-signal_frac-secpeak_frac), color='firebrick',
                     ls="--", label='background')

        if self._name_secpeak_pdf_ != "nosecpeak":
            secpeak_func = zfit.run(self._secpeak_pdf_.pdf(x_plot, norm_range=obs))
            plt.plot(x_plot, secpeak_func * norm * secpeak_frac, color='teal')
            plt.fill_between(x_plot, secpeak_func * norm * secpeak_frac, color='teal',
                            alpha=0.5, label='second signal')

        plt.plot(x_plot, signal_func * norm * signal_frac, color='seagreen')
        plt.fill_between(x_plot, signal_func * norm * signal_frac, color='seagreen',
                         alpha=0.5, label='signal')
        plt.plot(x_plot, total_func * norm, color='xkcd:blue', label='total fit')
        plt.xlabel(self._data_handler_.get_var_name())
        plt.xlim(limits[0], limits[1])
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

    def get_raw_yield(self):
        """
        Get the raw yield and its error

        Returns
        -------------------------------------------------
        raw_yield: float
            The raw yield obtained from the fit
        raw_yield_err: float
            The raw yield error obtained from the fit
        """
        return self._rawyield_, self._rawyield_err_

    def get_mass(self):
        """
        Get the mass and its error

        Returns
        -------------------------------------------------
        mass: float
            The mass value obtained from the fit
        mass_err: float
            The mass error obtained from the fit
        """
        return self._fit_result_.params['mass']['value'], self._fit_result_.params['mass']['hesse']['error']

    def get_width(self):
        """
        Get the width and its error

        Returns
        -------------------------------------------------
        width: float
            The width value obtained from the fit
        width_err: float
            The width error obtained from the fit
        """
        return self._fit_result_.params['width']['value'], self._fit_result_.params['width']['hesse']['error']

    def get_parameter(self, parameter):
        """
        Get the width and its error

        Parameters
        -------------------------------------------------
        parameter: str
            parameter to return

        Returns
        -------------------------------------------------
        parameter: float
            The parameter value obtained from the fit
        parameter_err: float
            The parameter error obtained from the fit

        """
        return self._fit_result_.params[parameter]['value'], self._fit_result_.params[parameter]['hesse']['error']

    def set_secpeak(self, pdf, mass, width):
        """
        Enable second peak and set the second peak initial parameters
        Parameters
        -------------------------------------------------
        pdf: str
            pdf to be used for the second peak. The possible options are:
            - 'gaussian'
            - 'crystalball'
        """

        self._name_secpeak_pdf_ = pdf
        self._init_mass_secpeak_ = mass
        self._init_width_secpeak_ = width
