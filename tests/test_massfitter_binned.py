"""
Test for binned fit with flarefly.F2MassFitter
"""

import os
import zfit
import uproot
import numpy as np
import matplotlib
from particle import Particle
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter

FITTERBINNEDDPLUS, FITTERBINNEDDSTAR, FITRES, FIG, RAWYHIST, RAWYIN = ([] for _ in range(6))
SGNPDFSDPLUS = ["crystalball", "gaussian", "doublegaus", "doublecb"]
BKGPDFSDPLUS = ["chebpol1", "expo"]
SGNPDFSDSTAR = ["gaussian", "voigtian"]
BKGPDFSDSTAR = ["expopow", "powlaw"]

# test all possible functions with D+
INFILEDPLUS = os.path.join(os.getcwd(), "tests/histos_dplus.root")
DATABINNEDDPLUS = DataHandler(INFILEDPLUS, var_name=r"$M_\mathrm{K\pi\pi}$ (GeV/$c^{2}$)",
                              histoname="hMass_80_120", limits=[1.75, 2.06], rebin=2)
for bkg_pdf in BKGPDFSDPLUS:
    for sgn_pdf in SGNPDFSDPLUS:
        FITTERBINNEDDPLUS.append(F2MassFitter(DATABINNEDDPLUS,
                                 name_signal_pdf=[sgn_pdf, "gaussian"],
                                 name_background_pdf=[bkg_pdf],
                                 name=f"{bkg_pdf}_{sgn_pdf}"))
        FITTERBINNEDDPLUS[-1].set_particle_mass(0, mass=1.872, fix=True)
        FITTERBINNEDDPLUS[-1].set_particle_mass(1, pdg_id=413, limits=[2.000, 2.020])
        FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "frac", 0.3, limits=[0.2, 0.4])
        FITTERBINNEDDPLUS[-1].set_signal_initpar(1, "frac", 0.05, limits=[0.01, 0.1])
        if sgn_pdf in ["gaussian", "crystalball"]:
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigma", 0.010)
        elif sgn_pdf == "doublecb":
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigma", 0.010)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "alphal", 2.)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "nl", 1.)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "alphar", 2.)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "nr", 1.)
        elif sgn_pdf == "doublegaus":
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigma1", 0.010)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigma2", 0.100)
        FITTERBINNEDDPLUS[-1].set_signal_initpar(1, "sigma", 0.015, limits=[0.01, 0.03])
        if bkg_pdf == "chebpol1":
            FITTERBINNEDDPLUS[-1].set_background_initpar(0, "c0", 2.)
            FITTERBINNEDDPLUS[-1].set_background_initpar(0, "c1", -0.5)
        FITRES.append(FITTERBINNEDDPLUS[-1].mass_zfit())
        FIG.append(FITTERBINNEDDPLUS[-1].plot_mass_fit(style="ATLAS"))
        RAWYHIST.append(uproot.open(INFILEDPLUS)["hRawYields"].to_numpy())
        RAWYIN.append(RAWYHIST[-1][0][4])

# test also bkg functions for D*
INFILEDSTAR = os.path.join(os.getcwd(), "tests/histos_dstar.root")
DATABINNEDDSTAR = DataHandler(INFILEDSTAR, var_name=r"$M_\mathrm{K\pi\pi}-M_\mathrm{K\pi}$ (GeV/$c^{2}$)",
                              histoname="hMass_40_60", limits=[Particle.from_pdgid(211).mass*1e-3, 0.155])
for bkg_pdf in BKGPDFSDSTAR:
    for sgn_pdf in SGNPDFSDSTAR:
        FITTERBINNEDDSTAR.append(F2MassFitter(DATABINNEDDSTAR,
                                        name_signal_pdf=[sgn_pdf],
                                        name_background_pdf=[bkg_pdf],
                                        name=f"{bkg_pdf}_{sgn_pdf}"))
        FITTERBINNEDDSTAR[-1].set_particle_mass(0, mass=0.1455, fix=True)
        if sgn_pdf == "gaussian":
            FITTERBINNEDDSTAR[-1].set_signal_initpar(0, "sigma", 0.007)
        elif sgn_pdf == "voigtian":
            FITTERBINNEDDSTAR[-1].set_signal_initpar(0, "sigma", 0.007)
            FITTERBINNEDDSTAR[-1].set_signal_initpar(0, "gamma", 70.e-6) # 70 keV
        FITRES.append(FITTERBINNEDDSTAR[-1].mass_zfit())
        FIG.append(FITTERBINNEDDSTAR[-1].plot_mass_fit(style="ATLAS"))
        RAWYHIST.append(uproot.open(INFILEDSTAR)["hRawYields"].to_numpy())
        RAWYIN.append(RAWYHIST[-1][0][1])

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
    for _, (fitterbinned, raw_in) in enumerate(zip(FITTERBINNEDDPLUS, RAWYIN)):
        rawy, rawy_err = fitterbinned.get_raw_yield()
        assert np.isclose(raw_in, rawy, atol=5*rawy_err)

def test_plot():
    """
    Test the mass fitter plot
    """
    for fig in FIG:
        assert isinstance(fig, matplotlib.figure.Figure)
