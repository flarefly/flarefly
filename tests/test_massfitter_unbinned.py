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
from flarefly import DataHandler, F2MassFitter
zfit.settings.set_seed(seed=42)


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
    TRUNCATED_NO_SGN = 8
    KDE = 9


# pylint: disable=duplicate-code
def create_sample(func_name, is_signal, size, **kwargs):
    """
    Create a sample from a given pdf function
    """
    fitter = F2MassFitter(DUMMY_DATA,
                          name_signal_pdf=[func_name] if is_signal else ['nosignal'],
                          name_background_pdf=[func_name] if not is_signal else ['nobkg'])
    for key, value in kwargs.items():
        if is_signal:
            fitter.set_signal_initpar(0, key, value)
        else:
            fitter.set_background_initpar(0, key, value)
    return fitter.sample_pdf(size)


LIMITS = [1.75, 2.1]
DUMMY_DATA = DataHandler(np.array(LIMITS), var_name='x', limits=LIMITS)
DATASGN = create_sample('gaussian', True, 10000, mu=1.865, sigma=0.010)
DATASGN2 = create_sample('gaussian', True, 20000, mu=1.968, sigma=0.010)
DATABKG = create_sample('expo', False, 30000, lam=-10)
LIFETIMEDATASGN = create_sample('expo', False, 10000, lam=-3)  # For sweights test
LIFETIMEDATASGN2 = create_sample('gaussian', True, 20000, mu=1.9, sigma=0.1)  # For sweights test
LIFETIMEDATABKG = create_sample('expo', False, 30000, lam=10)  # For sweights test
DATA = DataHandler(np.concatenate((DATASGN, DATABKG), axis=0),
                   var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITRES, FITTER, FIGS, RESIDUAL_FIGS = [], [], [], []
FITRES, FITTER, FIGS, RESIDUAL_FIGS = [], [], [], []
FITTER.append(F2MassFitter(DATA, name_signal_pdf=['gaussian'],
                           name_background_pdf=['expo'],
                           minuit_mode=1))
FITTER[TestCases.GAUS_EXPO].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITRES.append(FITTER[TestCases.GAUS_EXPO].mass_zfit(True, prefit_excluded_regions=[1.8, 1.95]))
FIGS.append(FITTER[TestCases.GAUS_EXPO].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.GAUS_EXPO].plot_raw_residuals(figsize=(10, 10)))
FITTER[TestCases.GAUS_EXPO].dump_to_root("test.root", num=100)

# Also consider extended case
FITTER.append(F2MassFitter(DATA, name_signal_pdf=['gaussian'],
                           name_background_pdf=['expo'],
                           minuit_mode=1,
                           extended=True))
FITTER[TestCases.GAUS_EXPO_EXTENDED].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITRES.append(FITTER[TestCases.GAUS_EXPO_EXTENDED].mass_zfit(True, prefit_excluded_regions=[1.8, 1.95]))
FIGS.append(FITTER[TestCases.GAUS_EXPO_EXTENDED].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.GAUS_EXPO_EXTENDED].plot_raw_residuals(figsize=(10, 10)))

# test also nobkg case
DATANOBKG = DataHandler(DATASGN, var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATANOBKG, name_signal_pdf=['gaussian'],
                           name_background_pdf=['nobkg'],
                           minuit_mode=1,
                           name="nobkg"))
FITRES.append(FITTER[TestCases.NOBKG].mass_zfit())
FIGS.append(FITTER[TestCases.NOBKG].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.NOBKG].plot_raw_residuals(figsize=(10, 10)))

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
FIGS.append(FITTER[TestCases.NOBKG_2SIGNAL].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.NOBKG_2SIGNAL].plot_raw_residuals(figsize=(10, 10)))

# no background, extended
FITTER.append(F2MassFitter(DATANOBKG, name_signal_pdf=['gaussian'],
                           name_background_pdf=['nobkg'],
                           minuit_mode=1,
                           extended=True))
FITRES.append(FITTER[TestCases.NOBKG_EXTENDED].mass_zfit(True, prefit_excluded_regions=[1.8, 1.95]))
FIGS.append(FITTER[TestCases.NOBKG_EXTENDED].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.NOBKG_EXTENDED].plot_raw_residuals(figsize=(10, 10)))

# test also nosignal case
DATANOSGN = DataHandler(DATABKG, var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATANOSGN, name_signal_pdf=['nosignal'],
                           name_background_pdf=['expo'],
                           minuit_mode=1,
                           name="nosignal"))
FITRES.append(FITTER[TestCases.NOSGN].mass_zfit())
FIGS.append(FITTER[TestCases.NOSGN].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.NOSGN].plot_raw_residuals(figsize=(10, 10)))

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
FIGS.append(FITTER[TestCases.FIX_FRAC].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.FIX_FRAC].plot_raw_residuals(figsize=(10, 10)))

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
FIGS.append(FITTER[TestCases.TRUNCATED].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.TRUNCATED].plot_raw_residuals(figsize=(10, 10)))

FITTER.append(F2MassFitter(DATA2, name_signal_pdf=['nosignal'],
                           name_background_pdf=['expo'],
                           limits=[[1.75, 1.92], [2.02, 2.1]],
                           minuit_mode=1,
                           name="truncated_bkg_only"))
FITTER[TestCases.TRUNCATED_NO_SGN].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITRES.append(FITTER[TestCases.TRUNCATED_NO_SGN].mass_zfit())
FIGS.append(FITTER[TestCases.TRUNCATED_NO_SGN].plot_mass_fit(figsize=(10, 10), show_extra_info=True))
RESIDUAL_FIGS.append(FITTER[TestCases.TRUNCATED_NO_SGN].plot_raw_residuals(figsize=(10, 10)))

# test kde
FITTER.append(F2MassFitter(DATA2, name_signal_pdf=['gaussian', 'kde_grid'],
                            name_background_pdf=["expo"],
                            minuit_mode=1,
                            name="kde"))
FITTER[TestCases.KDE].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITTER[TestCases.KDE].set_particle_mass(0, mass=1.85, limits=[1.84, 1.88])
FITTER[TestCases.KDE].set_signal_initpar(0, 'sigma', 0.01, limits=[0.005, 0.03])
data_hdl_kde = DataHandler(DATASGN2,
                           var_name=r"$M$ (GeV/$c^{2}$)",
                           limits=LIMITS)
FITTER[TestCases.KDE].set_signal_kde(1, data_hdl_kde)
FITRES.append(FITTER[TestCases.KDE].mass_zfit())
FIGS.append(FITTER[TestCases.KDE].plot_mass_fit(figsize=(10, 10)))
RESIDUAL_FIGS.append(FITTER[TestCases.KDE].plot_raw_residuals(figsize=(10, 10)))


def test_fitter():
    """
    Test the mass fitter
    """
    for res in FITRES:
        assert isinstance(res, zfit.minimizers.fitresult.FitResult)
    for i_fit, fit in enumerate(FITTER):
        assert isinstance(fit.get_name_signal_pdf(), list)
        assert isinstance(fit.get_name_background_pdf(), list)
        assert isinstance(fit.get_name_refl_pdf(), list)

        assert isinstance(fit.get_signal_pars(), list)
        assert isinstance(fit.get_signal_pars_uncs(), list)
        assert isinstance(fit.get_bkg_pars(), list)
        assert isinstance(fit.get_bkg_pars_uncs(), list)

        test_case = TestCases(i_fit)
        if test_case not in (TestCases.NOSGN, TestCases.TRUNCATED_NO_SGN):
            assert isinstance(fit.get_signal_pars()[0], dict)
            assert "sigma" in fit.get_signal_pars()[0]
            assert isinstance(fit.get_signal_pars_uncs()[0], dict)
            assert "sigma" in fit.get_signal_pars_uncs()[0]

        if test_case not in (TestCases.NOBKG, TestCases.NOBKG_2SIGNAL, TestCases.NOBKG_EXTENDED):
            assert isinstance(fit.get_bkg_pars()[0], dict)
            assert "lam" in fit.get_bkg_pars()[0]
            assert isinstance(fit.get_bkg_pars_uncs()[0], dict)
            assert "lam" in fit.get_bkg_pars_uncs()[0]

        assert isinstance(fit.sample_pdf(1000), np.ndarray)


def test_fitter_result():
    """
    Test the fitter output
    """
    for i_fit, fit in enumerate(FITTER):
        test_case = TestCases(i_fit)
        if test_case not in (TestCases.NOSGN, TestCases.TRUNCATED_NO_SGN):
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
        if test_case not in (
            TestCases.NOBKG, TestCases.NOBKG_2SIGNAL,
            TestCases.NOBKG_EXTENDED, TestCases.TRUNCATED_NO_SGN
        ):
            bkg_fit, bkg_fit_err = fit.get_background(min=1.8, max=1.95)
            true_bkg = np.count_nonzero(DATABKG[(DATABKG > 1.8) & (DATABKG < 1.95)])
            if test_case is TestCases.NOSGN:
                atol = 200
            else:
                atol = 5*bkg_fit_err
            assert np.isclose(true_bkg, bkg_fit, atol=atol)
        if test_case not in (TestCases.TRUNCATED, TestCases.TRUNCATED_NO_SGN):
            chi2_ndf = fit.get_chi2_ndf()
            assert chi2_ndf < 2
        if test_case in (TestCases.GAUS_EXPO, TestCases.GAUS_EXPO_EXTENDED):
            sweights = fit.get_sweights()
            bin_edges = np.linspace(*LIMITS, 101)
            hist_sweights_sgn, _ = np.histogram(np.concatenate([LIFETIMEDATASGN, LIFETIMEDATABKG]), bins=bin_edges, weights=sweights["signal0"])
            hist_sweights_bkg, _ = np.histogram(np.concatenate([LIFETIMEDATASGN, LIFETIMEDATABKG]), bins=bin_edges, weights=sweights["bkg0"])
            hist_true_sgn, _ = np.histogram(LIFETIMEDATASGN, bins=bin_edges)
            hist_true_bkg, _ = np.histogram(LIFETIMEDATABKG, bins=bin_edges)

            # check that the shapes are roughly correct (chi2 test)
            chi2_sgn = np.sum((hist_sweights_sgn - hist_true_sgn)**2 / (hist_true_sgn + 1e-6))
            chi2_bkg = np.sum((hist_sweights_bkg - hist_true_bkg)**2 / (hist_true_bkg + 1e-6))
            assert chi2_sgn < 200
            assert chi2_bkg < 200

        if test_case == TestCases.FIX_FRAC:
            sweights = fit.get_sweights()
            bin_edges = np.linspace(*LIMITS, 101)
            hist_sweights_sgn, _ = np.histogram(np.concatenate([LIFETIMEDATASGN, LIFETIMEDATASGN2, LIFETIMEDATABKG]), bins=bin_edges, weights=sweights["signal0"])
            hist_sweights_sgn2, bins1 = np.histogram(np.concatenate([LIFETIMEDATASGN, LIFETIMEDATASGN2, LIFETIMEDATABKG]), bins=bin_edges, weights=sweights["signal1"])
            hist_sweights_bkg, _ = np.histogram(np.concatenate([LIFETIMEDATASGN, LIFETIMEDATASGN2, LIFETIMEDATABKG]), bins=bin_edges, weights=sweights["bkg0"])
            hist_true_sgn, _ = np.histogram(LIFETIMEDATASGN, bins=bin_edges)
            hist_true_sgn2, bins2 = np.histogram(LIFETIMEDATASGN2, bins=bin_edges)
            hist_true_bkg, _ = np.histogram(LIFETIMEDATABKG, bins=bin_edges)

            # check that the shapes are roughly correct (chi2 test)
            chi2_sgn = np.sum((hist_sweights_sgn - hist_true_sgn)**2 / (hist_true_sgn + 1e-6))
            chi2_sgn2 = np.sum((hist_sweights_sgn2 - hist_true_sgn2)**2 / (hist_true_sgn2 + 1e-6))
            chi2_bkg = np.sum((hist_sweights_bkg - hist_true_bkg)**2 / (hist_true_bkg + 1e-6))
            assert chi2_sgn < 200
            assert chi2_sgn2 < 200
            assert chi2_bkg < 200

def test_plot():
    """
    Test the mass fitter plot
    """
    for fig in FIGS:
        assert isinstance(fig[0], matplotlib.figure.Figure)
        assert isinstance(fig[1], matplotlib.figure.Axes)
    for fig in RESIDUAL_FIGS:
        assert isinstance(fig, matplotlib.figure.Figure)


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
