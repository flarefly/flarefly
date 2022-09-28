"""
Test for binned fit with flarefly.F2MassFitter
"""

import os
import zfit
import uproot
import numpy as np
import matplotlib
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter

INFILE = os.path.join(os.getcwd(), 'tests/histos.root')
DATABINNED = DataHandler(INFILE, var_name=r'$M_\mathrm{K\pi\pi}$ (GeV/$c^{2}$)',
                         histoname='hMass_80_120', limits=[1.75, 2.06])

FITTERBINNED = F2MassFitter(DATABINNED,
                            name_signal_pdf=['gaussian', 'gaussian'],
                            name_background_pdf=['expo'])
FITTERBINNED.set_particle_mass(0, mass=1.872, fix=True)
FITTERBINNED.set_particle_mass(1, pdg_id=413, limits=[2.000, 2.020])
FITTERBINNED.set_signal_initpar(0, 'sigma', 0.010)
FITTERBINNED.set_signal_initpar(1, 'sigma', 0.015, limits=[0.01, 0.03])
FITRES = FITTERBINNED.mass_zfit()
FIG = FITTERBINNED.plot_mass_fit(style='ATLAS')

RAWYHIST = uproot.open(INFILE)["hRawYields"].to_numpy()
RAWYIN = RAWYHIST[0][4]

def test_fitter():
    """
    Test the mass fitter
    """
    assert isinstance(FITRES, zfit.minimizers.fitresult.FitResult)

def test_fitter_result():
    """
    Test the fitter output
    """
    rawy, rawy_err = FITTERBINNED.get_raw_yield()
    assert np.isclose(RAWYIN, rawy, atol=5*rawy_err)

def test_plot():
    """
    Test the mass fitter plot
    """
    assert isinstance(FIG, matplotlib.figure.Figure)
