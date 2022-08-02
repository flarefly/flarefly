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
        self._fit_result_ = None
        self._mass_ = 1.865
        self._width_ = 0.010
        self._signal_frac_ = 0.1
        self._alpha_ = 0.5
        self._bkg_pars_ = [-0.1]
        self._minimizer_ = zfit.minimize.Minuit(verbosity=7)
        self._rawyield_ = 0.
        self._rawyield_err_ = 0.

        zfit.settings.advanced_warnings.all = False
        zfit.settings.changed_warnings.all = False

    def __build_total_pdf(self):
        """
        Helper function to compose the total pdf
        """

        obs = self._data_handler_.get_obs()

        # signal pdf
        if self._name_signal_pdf_ == 'gaussian':
            mass = zfit.Parameter('mass', self._mass_)
            width = zfit.Parameter('sigma', self._width_)
            self._signal_pdf_ = zfit.pdf.Gauss(obs=obs, mu=mass, sigma=width)
        elif self._name_signal_pdf_ == 'crystalball':
            mass = zfit.Parameter('mass', self._mass_)
            width = zfit.Parameter('width', self._width_)
            alpha = zfit.Parameter('alpha', self._alpha_)
            nsig = zfit.Parameter('nsig', self._alpha_)
            self._signal_pdf_ = zfit.pdf.CrystalBall(obs=obs, mu=mass, sigma=width, alpha=alpha, n=nsig)
        else:
            Logger('Signal pdf not supported', 'FATAL')

        # background pdf
        if self._name_background_pdf_ == 'nobkg':
            Logger('Performing fit with no bkg pdf', 'WARNING')
        elif self._name_background_pdf_ == 'expo':
            lambd = zfit.Parameter('lambda', self._bkg_pars_[0])
            self._background_pdf_ = zfit.pdf.Exponential(lambd, obs=obs)
        else:
            Logger('Background pdf not supported', 'FATAL')

        if self._background_pdf_:
            self._signal_frac_ = zfit.Parameter('sig_frac', 0.1, 0., 1.)

            self._total_pdf_ = zfit.pdf.SumPDF(
                [self._signal_pdf_, self._background_pdf_], [self._signal_frac_]
            )
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
        self._mass_ = mass

    def set_peak_width(self, width):
        """
        Set the expected particle peak width

        Parameters
        -------------------------------------------------
        width: float
            The width to be set
        """
        self._width_ = width

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
            Logger('Binned data not yet supported', 'FATAL')
        else:
            nll = zfit.loss.UnbinnedNLL(
                model=self._total_pdf_, data=self._data_handler_.get_data())  # loss
            self._fit_result_ = self._minimizer_.minimize(loss=nll)
            self._fit_result_.hesse()

            if self._background_pdf_:
                tot_num = len(self._data_handler_.get_data().to_pandas())
                self._rawyield_ = self._fit_result_.params['sig_frac']['value'] * tot_num
                self._rawyield_err_ = self._fit_result_.params['sig_frac']['hesse']['error'] * tot_num
            else:
                self._rawyield_ = len(self._data_handler_.get_data().to_pandas())
                self._rawyield_err_ = np.sqrt(self._rawyield_)

        return self._fit_result_

    def plot_mass_fit(self, style='LHCb2'):
        """
        Plot the mass fit

        Parameters
        -------------------------------------------------
        style: str
            style to be used (see https://github.com/scikit-hep/mplhep for more details)

        Returns
        -------------------------------------------------
        fig: matplotlib.figure.Figure
            figure containing the mass fit plot
        """
        mplhep.style.use(style)

        obs = self._data_handler_.get_obs()
        lower, upper = obs.limits
        data_np = zfit.run(self._data_handler_.get_data()[:, 0])

        fig = plt.figure(figsize=(7, 7))
        if not self._data_handler_.get_is_binned():
            bins = 100
            counts, bin_edges = np.histogram(data_np, bins, range=(lower[-1][0], upper[0][0]))
            mplhep.histplot((counts, bin_edges), yerr=True, color='black', histtype='errorbar')

        x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
        total_func = zfit.run(self._total_pdf_.pdf(x_plot, norm_range=obs))
        bkg_func = zfit.run(self._background_pdf_.pdf(x_plot, norm_range=obs))
        signal_func = zfit.run(self._signal_pdf_.pdf(x_plot, norm_range=obs))
        signal_frac = self._fit_result_.params['sig_frac']['value']
        plt.plot(x_plot, bkg_func * data_np.shape[0] / bins * obs.area() * (1-signal_frac), color='xkcd:red')
        plt.plot(x_plot, signal_func * data_np.shape[0] / bins * obs.area() * signal_frac, color='xkcd:green')
        plt.plot(x_plot, total_func * data_np.shape[0] / bins * obs.area(), color='xkcd:blue')
        plt.xlabel(self._data_handler_.get_var_name())
        plt.ylabel(rf'counts / {(upper[0][0]-lower[-1][0])/bins*1000:0.1f} MeV/$c^2$')

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
