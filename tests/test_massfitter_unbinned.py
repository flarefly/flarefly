"""
Test for unbinned fit with flarefly.F2MassFitter
"""

import os
import zfit
import numpy as np
import matplotlib
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter

LIMITS = [1.75, 2.0]
DATASGN = np.random.normal(1.865, 0.010, size=10000)
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
    for fit in FITTER:
        rawy, rawy_err = fit.get_raw_yield()
        rawy_bc, rawy_bc_err = fit.get_raw_yield_bincounting()
        assert np.isclose(10000, rawy, atol=3*rawy_err)
        assert np.isclose(10000, rawy_bc, atol=3*rawy_bc_err)

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
