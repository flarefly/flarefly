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
SGNPDFSDPLUS = ["gaussian", "crystalball", "doublegaus", "doublecb"]
BKGPDFSDPLUS = ["expo", "chebpol1"]
SGNPDFSDSTAR = ["gaussian", "voigtian"]
BKGPDFSDSTAR = ["expopow", "powlaw"]

# test all possible functions with D+
INFILEDPLUS = os.path.join(os.getcwd(), "tests/histos_dplus.root")
DATABINNEDDPLUS = DataHandler(INFILEDPLUS, var_name=r"$M_\mathrm{K\pi\pi}$ (GeV/$c^{2}$)",
                              histoname="hMass_80_120", limits=[1.75, 2.06], rebin=4)
for bkg_pdf in BKGPDFSDPLUS:
    for sgn_pdf in SGNPDFSDPLUS:
        FITTERBINNEDDPLUS.append(F2MassFitter(DATABINNEDDPLUS,
                                 name_signal_pdf=[sgn_pdf, "gaussian"],
                                 name_background_pdf=[bkg_pdf],
                                 name=f"dplus_{bkg_pdf}_{sgn_pdf}",
                                 chi2_loss=True))
        FITTERBINNEDDPLUS[-1].set_particle_mass(0, mass=1.872, fix=True)
        FITTERBINNEDDPLUS[-1].set_particle_mass(1, pdg_id=413, limits=[2.00, 2.02])
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
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigma1", 0.010, limits=[0.008, 0.020])
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigma2", 0.100)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "frac1", 0.99, limits=[0.97, 1.00])
        FITTERBINNEDDPLUS[-1].set_signal_initpar(1, "sigma", 0.015, limits=[0.01, 0.03])
        if bkg_pdf == "chebpol1":
            FITTERBINNEDDPLUS[-1].set_background_initpar(0, "c0", -4.6)
            FITTERBINNEDDPLUS[-1].set_background_initpar(0, "c1", 1.6)
        elif bkg_pdf == "expo":
            FITTERBINNEDDPLUS[-1].set_background_initpar(0, "lam", -1.46)
        FITRES.append(FITTERBINNEDDPLUS[-1].mass_zfit())
        if FITRES[-1].converged:
            FIG.append(FITTERBINNEDDPLUS[-1].plot_mass_fit(style="ATLAS"))
            RAWYHIST.append(uproot.open(INFILEDPLUS)["hRawYields"].to_numpy())
            RAWYIN.append(RAWYHIST[-1][0][4])
        else:
            FIG.append(None)
            RAWYHIST.append(None)
            RAWYIN.append(None)

# test also bkg functions for D*
INFILEDSTAR = os.path.join(os.getcwd(), "tests/histos_dstar.root")
DATABINNEDDSTAR = DataHandler(INFILEDSTAR, var_name=r"$M_\mathrm{K\pi\pi}-M_\mathrm{K\pi}$ (GeV/$c^{2}$)",
                              histoname="hMass_40_60", limits=[Particle.from_pdgid(211).mass*1e-3, 0.155], rebin=4)
for bkg_pdf in BKGPDFSDSTAR:
    for sgn_pdf in SGNPDFSDSTAR:
        FITTERBINNEDDSTAR.append(F2MassFitter(DATABINNEDDSTAR,
                                        name_signal_pdf=[sgn_pdf],
                                        name_background_pdf=[bkg_pdf],
                                        name=f"dstar_{bkg_pdf}_{sgn_pdf}"))
        FITTERBINNEDDSTAR[-1].set_particle_mass(0, mass=0.1455, fix=True)
        if sgn_pdf == "gaussian":
            FITTERBINNEDDSTAR[-1].set_signal_initpar(0, "sigma", 0.007)
        elif sgn_pdf == "voigtian":
            FITTERBINNEDDSTAR[-1].set_signal_initpar(0, "sigma", 0.007)
            FITTERBINNEDDSTAR[-1].set_signal_initpar(0, "gamma", 70.e-6) # 70 keV
        FITRES.append(FITTERBINNEDDSTAR[-1].mass_zfit())
        if FITRES[-1].converged:
            FIG.append(FITTERBINNEDDSTAR[-1].plot_mass_fit(style="ATLAS"))
            RAWYHIST.append(uproot.open(INFILEDSTAR)["hRawYields"].to_numpy())
            RAWYIN.append(RAWYHIST[-1][0][1])
        else:
            FIG.append(None)
            RAWYHIST.append(None)
            RAWYIN.append(None)

# test also D0 reflections
INFILED0 = os.path.join(os.getcwd(), "tests/histos_dzero.root")
DATABINNEDD0 = DataHandler(INFILED0, var_name=r"$M_\mathrm{K\pi}$ (GeV/$c^{2}$)",
                           histoname="histMass_6", limits=[1.7, 2.10], rebin=4)
REFLBINNEDD0 = DataHandler(INFILED0, var_name=r"$M_\mathrm{K\pi}$ (GeV/$c^{2}$)",
                           histoname="histRfl_6", limits=[1.7, 2.10], rebin=4)

FITTERD0 = F2MassFitter(DATABINNEDD0, name_signal_pdf=["gaussian"],
                        name_background_pdf=["chebpol2"],
                        name_refl_pdf=["hist"], name="dzero")
FITTERD0.set_reflection_template(0, REFLBINNEDD0, 0.1)
FITTERD0.set_particle_mass(0, pdg_id=421, fix=False)
FITTERD0.set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])
FITTERD0.set_signal_initpar(0, "sigma", 0.01, limits=[0.005, 0.03])
FITTERD0.set_background_initpar(0, "c0", 2.)
FITTERD0.set_background_initpar(0, "c1", -0.293708)
FITTERD0.set_background_initpar(0, "c2", 0.02)
FITRES.append(FITTERD0.mass_zfit())
if FITRES[-1].converged:
    FIG.append(FITTERBINNEDDPLUS[-1].plot_mass_fit(style="ATLAS"))
    RAWYHIST.append(None) # we do not have it, to be fixed
    RAWYIN.append(None) # we do not have it, to be fixed
else:
    FIG.append(None)
    RAWYHIST.append(None)
    RAWYIN.append(None)

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
