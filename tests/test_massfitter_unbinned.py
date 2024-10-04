"""
Test for unbinned fit with flarefly.F2MassFitter
"""

import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1" # pylint: disable=wrong-import-position
import zfit
import numpy as np
import matplotlib
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter

LIMITS = [1.75, 2.1]
DATASGN = np.random.normal(1.865, 0.010, size=10000)
DATASGN2 = np.random.normal(1.968, 0.010, size=20000)
DATABKG = np.random.uniform(LIMITS[0], LIMITS[1], size=10000)
DATA = DataHandler(np.concatenate((DATASGN, DATABKG), axis=0),
                   var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)

FITRES, FITTER, FIGS = [], [], []
FITTER.append(F2MassFitter(DATA, name_signal_pdf=['gaussian'],
                           name_background_pdf=['expo'],
                           minuit_mode=1))
FITTER[0].set_background_initpar(0, 'lam', 0.1, limits=[-10., 10.], fix=False)
FITRES.append(FITTER[0].mass_zfit())
FIGS.append(FITTER[0].plot_mass_fit(figsize=(10, 10)))
FITTER[0].dump_to_root("test.root")

# test also nobkg case
DATANOBKG = DataHandler(DATASGN, var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATANOBKG, name_signal_pdf=['gaussian'],
                           name_background_pdf=['nobkg'],
                           minuit_mode=1,
                           name="nobkg"))
FITRES.append(FITTER[1].mass_zfit())
FIGS.append(FITTER[1].plot_mass_fit(figsize=(10, 10)))

# test fixing the relative pdf fractions
DATA2 = DataHandler(np.concatenate((DATASGN, DATASGN2, DATABKG), axis=0),
                    var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
FITTER.append(F2MassFitter(DATA2, name_signal_pdf=['gaussian', 'gaussian'],
                           name_background_pdf=['chebpol1'],
                           minuit_mode=1))
FITTER[2].set_particle_mass(0, mass=1.8)
FITTER[2].set_particle_mass(1, mass=2.0)
FITTER[2].set_signal_initpar(0, 'sigma', 0.05)
FITTER[2].set_signal_initpar(1, 'sigma', 0.05)
FITTER[2].fix_signal_frac_to_signal_pdf(1, 0, 2)
FITRES.append(FITTER[2].mass_zfit())
FIGS.append(FITTER[2].plot_mass_fit(figsize=(10, 10)))

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
        if i_fit == 2:
            rawy2, rawy2_err = fit.get_raw_yield(1)
            rawy2_bc, rawy2_bc_err = fit.get_raw_yield_bincounting(1)
            assert np.isclose(20000, rawy2, atol=3*rawy2_err)
            assert np.isclose(20000, rawy2_bc, atol=3*rawy2_bc_err)

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
    os.remove("test.root")
