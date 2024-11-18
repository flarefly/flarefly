"""
Test for unbinned fit with flarefly.F2MassFitter
"""

import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"  # pylint: disable=wrong-import-position
import zfit
import uproot
import numpy as np
import matplotlib
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter

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
FITTER[0].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITRES.append(FITTER[0].mass_zfit(True, prefit_excluded_regions=[1.8, 1.95]))
FIGS.append(FITTER[0].plot_mass_fit(figsize=(10, 10)))
FITTER[0].dump_to_root("test.root", num=100)

# test also nobkg case
DATANOBKG = DataHandler(DATASGN, var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATANOBKG, name_signal_pdf=['gaussian'],
                           name_background_pdf=['nobkg'],
                           minuit_mode=1,
                           name="nobkg"))
FITRES.append(FITTER[1].mass_zfit())
FIGS.append(FITTER[1].plot_mass_fit(figsize=(10, 10)))

DATANOBKG2 = DataHandler(np.concatenate((DATASGN, DATASGN2), axis=0),
                         var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATANOBKG2,
                           name_signal_pdf=['gaussian', 'gaussian'],
                           name_background_pdf=['nobkg'],
                           minuit_mode=1,
                           name="nobkg_2signals"))
FITTER[2].set_signal_initpar(0, 'sigma', 0.01, limits=[0., 1.e6])
FITTER[2].set_signal_initpar(1, 'sigma', 0.01, limits=[0., 1.e6])
FITRES.append(FITTER[2].mass_zfit())
FIGS.append(FITTER[2].plot_mass_fit(figsize=(10, 10)))

# test fixing the relative pdf fractions
DATA2 = DataHandler(np.concatenate((DATASGN, DATASGN2, DATABKG), axis=0),
                    var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATA2, name_signal_pdf=['gaussian', 'gaussian'],
                           name_background_pdf=['expo'],
                           minuit_mode=1,
                           name="fix_frac"))
FITTER[3].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITTER[3].set_particle_mass(0, mass=1.85, limits=[1.84, 1.88])
FITTER[3].set_particle_mass(1, mass=1.95, limits=[1.94, 1.98])
FITTER[3].set_signal_initpar(0, 'sigma', 0.01, limits=[0.005, 0.03])
FITTER[3].set_signal_initpar(1, 'sigma', 0.01, limits=[0.005, 0.03])
FITTER[3].fix_signal_frac_to_signal_pdf(1, 0, 2)
FITRES.append(FITTER[3].mass_zfit(True, prefit_exclude_nsigma=5.))
FIGS.append(FITTER[3].plot_mass_fit(figsize=(10, 10)))

# test truncated pdfs
FITTER.append(F2MassFitter(DATA2, name_signal_pdf=['gaussian'],
                           name_background_pdf=['expo'],
                           limits=[[1.75, 1.92], [2.02, 2.1]],
                           minuit_mode=1,
                           name="truncated"))
FITTER[4].set_background_initpar(0, 'lam', -10, limits=[-15., -5.], fix=False)
FITTER[4].set_particle_mass(0, mass=1.85, limits=[1.84, 1.88])
FITTER[4].set_signal_initpar(0, 'sigma', 0.01, limits=[0.005, 0.03])
FITRES.append(FITTER[4].mass_zfit())
FIGS.append(FITTER[4].plot_mass_fit(figsize=(10, 10)))


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
        rawy, rawy_err = fit.get_raw_yield()
        rawy_bc, rawy_bc_err = fit.get_raw_yield_bincounting()
        assert np.isclose(10000, rawy, atol=3*rawy_err)
        assert np.isclose(10000, rawy_bc, atol=3*rawy_bc_err)
        if i_fit in (2, 3):  # test the case with two signals
            rawy2, rawy2_err = fit.get_raw_yield(1)
            rawy2_bc, rawy2_bc_err = fit.get_raw_yield_bincounting(1)
            assert np.isclose(20000, rawy2, atol=3*rawy2_err)
            assert np.isclose(20000, rawy2_bc, atol=3*rawy2_bc_err)
        if i_fit not in (1, 2):  # test the cases with background
            bkg_fit, bkg_fit_err = fit.get_background(min=1.8, max=1.95)
            true_bkg = np.count_nonzero(DATABKG[(DATABKG > 1.8) & (DATABKG < 1.95)])
            assert np.isclose(true_bkg, bkg_fit, atol=5*bkg_fit_err)
        if i_fit != 4:
            chi2_ndf = fit.get_chi2_ndf()
            assert chi2_ndf < 2


def test_plot():
    """
    Test the mass fitter plot
    """
    for fig in FIGS:
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
