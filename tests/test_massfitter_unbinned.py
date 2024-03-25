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

FITTER = F2MassFitter(DATA, name_signal_pdf=['gaussian'],
                      name_background_pdf=['expo'],
                      minuit_mode=1)
FITTER.set_background_initpar(0, 'lam', 0.1, limits=[-10., 10.], fix=False)
FITRES = FITTER.mass_zfit()
FIG = FITTER.plot_mass_fit(figsize=(10, 10))
FITTER.dump_to_root("test.root")

def test_fitter():
    """
    Test the mass fitter
    """
    assert isinstance(FITRES, zfit.minimizers.fitresult.FitResult)

def test_fitter_result():
    """
    Test the fitter output
    """
    rawy, rawy_err = FITTER.get_raw_yield()
    assert np.isclose(10000, rawy, atol=3*rawy_err)

def test_plot():
    """
    Test the mass fitter plot
    """
    assert isinstance(FIG, matplotlib.figure.Figure)

def test_dump():
    """
    Test the dump of the root file
    """

    assert os.path.isfile("test.root")
    os.remove("test.root")
