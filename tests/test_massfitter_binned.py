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

bkg_pdfs = ['chebpol1', 'expo']
FITTERBINNED, FITRES, FIG, RAWYHIST, RAWYIN = ([] for _ in range(5))
for ibkg, bkg_pdf in enumerate(bkg_pdfs):
    FITTERBINNED.append(F2MassFitter(DATABINNED,
                                     name_signal_pdf=['gaussian', 'gaussian'],
                                     name_background_pdf=[bkg_pdf],
                                     name=f"{bkg_pdf}"))
    FITTERBINNED[ibkg].set_particle_mass(0, mass=1.872, fix=True)
    FITTERBINNED[ibkg].set_particle_mass(1, pdg_id=413, limits=[2.000, 2.020])
    FITTERBINNED[ibkg].set_signal_initpar(0, 'sigma', 0.010)
    FITTERBINNED[ibkg].set_signal_initpar(1, 'sigma', 0.015, limits=[0.01, 0.03])
    if bkg_pdf == 'chebpol1':
        FITTERBINNED[ibkg].set_background_initpar(0, 'c0', 850)
        FITTERBINNED[ibkg].set_background_initpar(0, 'c1', -0.4)
    FITRES.append(FITTERBINNED[ibkg].mass_zfit())
    FIG.append(FITTERBINNED[ibkg].plot_mass_fit(style='ATLAS'))
    RAWYHIST.append(uproot.open(INFILE)["hRawYields"].to_numpy())
    RAWYIN.append(RAWYHIST[ibkg][0][4])

def test_fitter():
    """
    Test the mass fitter
    """
    for fitres in FITRES:
        assert isinstance(fitres, zfit.minimizers.fitresult.FitResult)

def test_fitter_result():
    """
    Test the fitter output
    """
    for _, (fitterbinned, raw_in) in enumerate(zip(FITTERBINNED, RAWYIN)):
        rawy, rawy_err = fitterbinned.get_raw_yield()
        assert np.isclose(raw_in, rawy, atol=5*rawy_err)

def test_plot():
    """
    Test the mass fitter plot
    """
    for fig in FIG:
        assert isinstance(fig, matplotlib.figure.Figure)
