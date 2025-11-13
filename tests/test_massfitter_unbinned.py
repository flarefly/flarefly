"""
Test for unbinned fit with flarefly.F2MassFitter
"""

from enum import IntEnum
import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"  # pylint: disable=wrong-import-position
import zfit
import uproot
import numpy as np
import matplotlib
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter


class TestCases(IntEnum):
    """
    Test enum
    """
    __test__ = False

    GAUS_EXPO = 0
    GAUS_EXPO_EXTENDED = 1
    NOBKG = 2
    NOBKG_2SIGNAL = 3
    NOBKG_EXTENDED = 4
    NOSGN = 5
    FIX_FRAC = 6
    TRUNCATED = 7


LIMITS = [1.75, 2.1]
np.random.seed(42)
DATASGN = np.random.normal(1.865, 0.010, size=10000)
DATASGN2 = np.random.normal(1.968, 0.010, size=20000)
DATABKG = np.random.exponential(scale=0.1, size=30000) + LIMITS[0]
DATA = DataHandler(np.concatenate((DATASGN, DATABKG), axis=0),
                   var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)

FITRES, FITTER, FIGS = [], [], []
FITTER.append(F2MassFitter(DATA, name_signal_pdf=['gaussian'],
                           name_background_pdf=['expo'],
                           minuit_mode=1))
FITTER[TestCases.GAUS_EXPO].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITRES.append(FITTER[TestCases.GAUS_EXPO].mass_zfit(True, prefit_excluded_regions=[1.8, 1.95]))
FIGS.append(FITTER[TestCases.GAUS_EXPO].plot_mass_fit(figsize=(10, 10)))
FITTER[TestCases.GAUS_EXPO].dump_to_root("test.root", num=100)

# Also consider extended case
FITTER.append(F2MassFitter(DATA, name_signal_pdf=['gaussian'],
                           name_background_pdf=['expo'],
                           minuit_mode=1,
                           extended=True))
FITTER[TestCases.GAUS_EXPO_EXTENDED].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITRES.append(FITTER[TestCases.GAUS_EXPO_EXTENDED].mass_zfit(True, prefit_excluded_regions=[1.8, 1.95]))
FIGS.append(FITTER[TestCases.GAUS_EXPO_EXTENDED].plot_mass_fit(figsize=(10, 10)))

# test also nobkg case
DATANOBKG = DataHandler(DATASGN, var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATANOBKG, name_signal_pdf=['gaussian'],
                           name_background_pdf=['nobkg'],
                           minuit_mode=1,
                           name="nobkg"))
FITRES.append(FITTER[TestCases.NOBKG].mass_zfit())
FIGS.append(FITTER[TestCases.NOBKG].plot_mass_fit(figsize=(10, 10)))

# no background, two signals
DATANOBKG2 = DataHandler(np.concatenate((DATASGN, DATASGN2), axis=0),
                         var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATANOBKG2,
                           name_signal_pdf=['gaussian', 'gaussian'],
                           name_background_pdf=['nobkg'],
                           minuit_mode=1,
                           name="nobkg_2signals"))
FITTER[TestCases.NOBKG_2SIGNAL].set_signal_initpar(0, 'sigma', 0.01, limits=[0., 1.e6])
FITTER[TestCases.NOBKG_2SIGNAL].set_signal_initpar(1, 'sigma', 0.01, limits=[0., 1.e6])
FITRES.append(FITTER[TestCases.NOBKG_2SIGNAL].mass_zfit())
FIGS.append(FITTER[TestCases.NOBKG_2SIGNAL].plot_mass_fit(figsize=(10, 10)))

# no background, extended
FITTER.append(F2MassFitter(DATANOBKG, name_signal_pdf=['gaussian'],
                           name_background_pdf=['nobkg'],
                           minuit_mode=1,
                           extended=True))
FITRES.append(FITTER[TestCases.NOBKG_EXTENDED].mass_zfit(True, prefit_excluded_regions=[1.8, 1.95]))
FIGS.append(FITTER[TestCases.NOBKG_EXTENDED].plot_mass_fit(figsize=(10, 10)))

# test also nosignal case
DATANOSGN = DataHandler(DATABKG, var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATANOSGN, name_signal_pdf=['nosignal'],
                           name_background_pdf=['expo'],
                           minuit_mode=1,
                           name="nosignal"))
FITRES.append(FITTER[TestCases.NOSGN].mass_zfit())
FIGS.append(FITTER[TestCases.NOSGN].plot_mass_fit(figsize=(10, 10)))

# test fixing the relative pdf fractions
DATA2 = DataHandler(np.concatenate((DATASGN, DATASGN2, DATABKG), axis=0),
                    var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATA2, name_signal_pdf=['gaussian', 'gaussian'],
                           name_background_pdf=['expo'],
                           minuit_mode=1,
                           name="fix_frac"))
FITTER[TestCases.FIX_FRAC].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITTER[TestCases.FIX_FRAC].set_particle_mass(0, mass=1.85, limits=[1.84, 1.88])
FITTER[TestCases.FIX_FRAC].set_particle_mass(1, mass=1.95, limits=[1.94, 1.98])
FITTER[TestCases.FIX_FRAC].set_signal_initpar(0, 'sigma', 0.01, limits=[0.005, 0.03])
FITTER[TestCases.FIX_FRAC].set_signal_initpar(1, 'sigma', 0.01, limits=[0.005, 0.03])
FITTER[TestCases.FIX_FRAC].fix_signal_frac_to_signal_pdf(1, 0, 2)
FITRES.append(FITTER[TestCases.FIX_FRAC].mass_zfit(True, prefit_exclude_nsigma=5.))
FIGS.append(FITTER[TestCases.FIX_FRAC].plot_mass_fit(figsize=(10, 10)))

# test truncated pdfs
FITTER.append(F2MassFitter(DATA2, name_signal_pdf=['gaussian'],
                           name_background_pdf=['expo'],
                           limits=[[1.75, 1.92], [2.02, 2.1]],
                           minuit_mode=1,
                           name="truncated"))
FITTER[TestCases.TRUNCATED].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITTER[TestCases.TRUNCATED].set_particle_mass(0, mass=1.85, limits=[1.84, 1.88])
FITTER[TestCases.TRUNCATED].set_signal_initpar(0, 'sigma', 0.01, limits=[0.005, 0.03])
FITRES.append(FITTER[TestCases.TRUNCATED].mass_zfit())
FIGS.append(FITTER[TestCases.TRUNCATED].plot_mass_fit(figsize=(10, 10)))


def test_fitter():
    """
    Test the mass fitter
    """
    for res in FITRES:
        assert isinstance(res, zfit.minimizers.fitresult.FitResult)


def test_fitter_result():
    """
    Test the fitter output
    """
    for i_fit, fit in enumerate(FITTER):
        test_case = TestCases(i_fit)
        if test_case != TestCases.NOSGN:
            rawy, rawy_err = fit.get_raw_yield()
            rawy_bc, rawy_bc_err = fit.get_raw_yield_bincounting()
            assert np.isclose(10000, rawy, atol=3*rawy_err)
            assert np.isclose(10000, rawy_bc, atol=3*rawy_bc_err)
        # test the case with two signals
        if test_case in (TestCases.NOBKG_2SIGNAL, TestCases.FIX_FRAC):
            rawy2, rawy2_err = fit.get_raw_yield(1)
            rawy2_bc, rawy2_bc_err = fit.get_raw_yield_bincounting(1)
            assert np.isclose(20000, rawy2, atol=3*rawy2_err)
            assert np.isclose(20000, rawy2_bc, atol=3*rawy2_bc_err)
        # test the cases with background
        if test_case not in (TestCases.NOBKG, TestCases.NOBKG_2SIGNAL, TestCases.NOBKG_EXTENDED):
            bkg_fit, bkg_fit_err = fit.get_background(min=1.8, max=1.95)
            true_bkg = np.count_nonzero(DATABKG[(DATABKG > 1.8) & (DATABKG < 1.95)])
            if test_case is TestCases.NOSGN:
                atol = 200
            else:
                atol = 5*bkg_fit_err
            assert np.isclose(true_bkg, bkg_fit, atol=atol)
        if test_case != TestCases.TRUNCATED:
            chi2_ndf = fit.get_chi2_ndf()
            assert chi2_ndf < 2


def test_plot():
    """
    Test the mass fitter plot
    """
    for ifig, fig in enumerate(FIGS):
        fig[0].savefig(f"plots/test{ifig}.png")
        assert isinstance(fig[0], matplotlib.figure.Figure)
        assert isinstance(fig[1], matplotlib.figure.Axes)


def test_dump():
    """
    Test the dump of the root file
    """

    assert os.path.isfile("test.root")
    with uproot.open("test.root", encoding='utf-8') as f:
        h_signal = f["signal_0"]
        h_bkg = f["bkg_0"]
        signal = sum(h_signal.values())
        bkg = sum(h_bkg.values())
        assert np.isclose(10000, signal, atol=2000)
        assert np.isclose(30000, bkg, atol=2000)
    os.remove("test.root")
