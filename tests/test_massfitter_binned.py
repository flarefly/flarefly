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

FITTERBINNEDDPLUS, FITTERBINNEDDSTAR, FITTERBINNEDD0, FITRES, FIG, RAWYHIST, RAWYIN = ([] for _ in range(7))
SGNPDFSDPLUS = ["gaussian", "crystalball", "doublegaus", "doublecb", "genercrystalball"]
BKGPDFSDPLUS = ["expo", "chebpol1"]
SGNPDFSDSTAR = ["gaussian", "voigtian"]
BKGPDFSDSTAR = ["expopow", "powlaw", "expopowext"]
SGNPDFSD0 = ["gaussian", "gausexptail", "genergausexptail"]
BKGPDFSD0 = ["expo"]

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
                                 chi2_loss=True, tol=1.e-3))
        FITTERBINNEDDPLUS[-1].set_particle_mass(0, mass=1.872, fix=True)
        FITTERBINNEDDPLUS[-1].set_particle_mass(1, pdg_id=413, limits=[2.00, 2.02])
        FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "frac", 0.3, limits=[0.2, 0.4])
        FITTERBINNEDDPLUS[-1].set_signal_initpar(1, "frac", 0.05, limits=[0.01, 0.1])
        if sgn_pdf in ["gaussian", "crystalball", "doublecb"]:
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigma", 0.010)
        if sgn_pdf in ["doublecb", "genercrystalball"]:
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "alphal", 2.)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "nl", 1.)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "alphar", 2.)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "nr", 1.)
        if sgn_pdf == "genercrystalball":
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigmal", 0.010)
            FITTERBINNEDDPLUS[-1].set_signal_initpar(0, "sigmar", 0.010)
        if sgn_pdf == "doublegaus":
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
                              histoname="hMass_40_60", limits=[Particle.from_pdgid(211).mass*1e-3, 0.155], rebin=4,
                              tol=1.e-3)
for bkg_pdf in BKGPDFSDSTAR:
    for sgn_pdf in SGNPDFSDSTAR:
        FITTERBINNEDDSTAR.append(F2MassFitter(DATABINNEDDSTAR,
                                        name_signal_pdf=[sgn_pdf],
                                        name_background_pdf=[bkg_pdf],
                                        name=f"dstar_{bkg_pdf}_{sgn_pdf}"))
        FITTERBINNEDDSTAR[-1].set_particle_mass(0, mass=0.1455, fix=True)
        if sgn_pdf in ["gaussian", "voigtian"]:
            FITTERBINNEDDSTAR[-1].set_signal_initpar(0, "sigma", 0.0007, limits=[0.0001, 0.0015])
        elif sgn_pdf == "voigtian":
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

for bkg_pdf in BKGPDFSD0:
    for sgn_pdf in SGNPDFSD0:
        FITTERBINNEDD0.append(F2MassFitter(DATABINNEDD0, name_signal_pdf=[sgn_pdf],
                              name_background_pdf=[bkg_pdf],
                              name_refl_pdf=["hist"], name=f"dzero_{bkg_pdf}_{sgn_pdf}", tol=1.e-3))
        FITTERBINNEDD0[-1].set_reflection_template(0, REFLBINNEDD0, 0.294)
        FITTERBINNEDD0[-1].set_particle_mass(0, pdg_id=421, limits=[1.86, 1.90])
        FITTERBINNEDD0[-1].set_signal_initpar(0, "frac", 0.1, limits=[0., 1.])
        FITTERBINNEDD0[-1].set_signal_initpar(0, "sigma", 0.01, limits=[0.005, 0.03])
        if sgn_pdf == "gausexptail":
            FITTERBINNEDD0[-1].set_signal_initpar(0, "alpha", 1.e6, limits=[0., 1.e10], fix=True)
        elif sgn_pdf == "genergausexptail":
            FITTERBINNEDD0[-1].set_signal_initpar(0, "alphar", 1.e6, fix=True)
            FITTERBINNEDD0[-1].set_signal_initpar(0, "alphal", 1.e6, fix=True)
            FITTERBINNEDD0[-1].set_signal_initpar(0, "sigmar", 0.01, limits=[0.005, 0.03])
            FITTERBINNEDD0[-1].set_signal_initpar(0, "sigmal", 0.01, limits=[0.005, 0.03])
        FITTERBINNEDD0[-1].set_background_initpar(0, "c0", 2.)
        FITTERBINNEDD0[-1].set_background_initpar(0, "c1", -0.293708)
        FITTERBINNEDD0[-1].set_background_initpar(0, "c2", 0.02)
        FITRES.append(FITTERBINNEDD0[-1].mass_zfit())
        if FITRES[-1].converged:
            FIG.append(FITTERBINNEDD0[-1].plot_mass_fit(style="ATLAS"))
            RAWYHIST.append(uproot.open(INFILED0)["hSignal"].to_numpy())
            RAWYIN.append(RAWYHIST[-1][0][2]) # we do not have it, to be fixed
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
    for ifit, (fitterbinned, raw_in) in enumerate(zip(FITTERBINNEDDPLUS+FITTERBINNEDDSTAR+FITTERBINNEDD0, RAWYIN)):
        rawy, rawy_err = fitterbinned.get_raw_yield()
        assert np.isclose(raw_in, rawy, atol=5*rawy_err)
        if ifit == 0: # only test bin counting with Gaussian functions for simplicity
            rawy_bc, rawy_bc_err = fitterbinned.get_raw_yield_bincounting()
            assert np.isclose(raw_in, rawy_bc, atol=5*rawy_bc_err)

def test_plot():
    """
    Test the mass fitter plot
    """
    for fig in FIG:
        assert isinstance(fig, matplotlib.figure.Figure)
