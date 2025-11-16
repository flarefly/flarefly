"""
Test for binned fit with flarefly.F2MassFitter
"""

import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"  # pylint: disable=wrong-import-position
import zfit
import numpy as np
import matplotlib
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
zfit.settings.set_seed(seed=42)


# pylint: disable=duplicate-code
def create_sample(func_name, is_signal, size, **kwargs):
    """
    Create a sample from a given pdf function
    """
    fitter = F2MassFitter(DUMMY_DATA,
                          name_signal_pdf=[func_name] if is_signal else ['nosignal'],
                          name_background_pdf=[func_name] if not is_signal else ['nobkg'])
    for k, v in kwargs.items():
        if is_signal:
            fitter.set_signal_initpar(0, k, v)
        else:
            fitter.set_background_initpar(0, k, v)
    return fitter.sample_pdf(size)


LIMITS = [0.15, 1]
DUMMY_DATA = DataHandler(np.array(LIMITS), var_name='x', limits=LIMITS)
SGN_PDFS_NAMES = [
    "gaussian", "crystalball", "doublegaus", "doublecb", "doublecbsymm",
    "genercrystalball", "gausexptail", "genergausexptail", "genergausexptailsymm"
]
BKG_PDFS_NAMES = ["expo", "chebpol2", "expopow", "powlaw", "expopowext", "powlaw", "expopowext"]
SGN_PARAMS = {
    "gaussian": {"mu": 0.4, "sigma": 0.03},
    "crystalball": {"mu": 0.4, "sigma": 0.03, "alpha": 1.5, "n": 3.},
    "doublegaus": {"mu": 0.4, "sigma1": 0.03, "sigma2": 0.07, "frac1": 0.7},
    "doublecb": {"mu": 0.4, "sigma": 0.03, "alphal": 1.5, "nl": 3., "alphar": 1.5, "nr": 3.},
    "doublecbsymm": {"mu": 0.4, "sigma": 0.03, "alpha": 1.5, "n": 3.},
    "genercrystalball": {"mu": 0.4, "sigmal": 0.03, "sigmar": 0.03, "alphal": 1.5, "nl": 3., "alphar": 1.5, "nr": 3.},
    "gausexptail": {"mu": 0.4, "sigma": 0.03, "alpha": 4},
    "genergausexptail": {"mu": 0.4, "sigmal": 0.03, "sigmar": 0.03, "alphal": 4, "alphar": 4},
    "genergausexptailsymm": {"mu": 0.4, "sigma": 0.03, "alpha": 4},
}
BKG_PARAMS = {
    "expo": {"lam": -6},
    "chebpol2": {"c0": 2, "c1": 1, "c2": 1},
    "expopow": {"lam": 0.001},
    "powlaw": {"power": 1.},
    "expopowext": {"power": 0.5, "c1": -0.1, "c2": 0., "c3": 0.0},
}
SGN_DATA = {
    sgn_pdf_name: create_sample(sgn_pdf_name, True, 10000, **SGN_PARAMS[sgn_pdf_name])
    for sgn_pdf_name in SGN_PDFS_NAMES
}
BKG_DATA = {
    bkg_pdf_name: create_sample(bkg_pdf_name, False, 30000, **BKG_PARAMS[bkg_pdf_name])
    for bkg_pdf_name in BKG_PDFS_NAMES
}
REFL_DATA = create_sample("gaussian", True, 5000, mu=0.4, sigma=0.06)
FITTERS, FITRES, FIGS, PDFS = [], [], [], []

for sgn_pdf_name, bkg_pdf_name in zip(
        [SGN_PDFS_NAMES[0]]*len(BKG_PDFS_NAMES) + SGN_PDFS_NAMES,
        BKG_PDFS_NAMES + [BKG_PDFS_NAMES[0]]*len(SGN_PDFS_NAMES)):
    # We do gaussian with all backgrounds and all signals with expo background
    DATA = DataHandler(
        np.concatenate((SGN_DATA[sgn_pdf_name], BKG_DATA[bkg_pdf_name]), axis=0),
        var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS, rebin=4
    )
    DATA = DATA.get_binned_data_handler_from_unbinned_data()
    FITTERS.append(F2MassFitter(
        DATA,
        name_signal_pdf=[sgn_pdf_name],
        name_background_pdf=[bkg_pdf_name],
        name=f"{sgn_pdf_name}_{bkg_pdf_name}",
        chi2_loss=True, tol=1.e-3
    ))
    for key, value in SGN_PARAMS[sgn_pdf_name].items():
        FITTERS[-1].set_signal_initpar(0, key, value)
    for key, value in BKG_PARAMS[bkg_pdf_name].items():
        FITTERS[-1].set_background_initpar(0, key, value)
    FITRES.append(FITTERS[-1].mass_zfit())
    FIGS.append(FITTERS[-1].plot_mass_fit(figsize=(10, 10)))
    PDFS.append((sgn_pdf_name, bkg_pdf_name))

# Reflection test
DATA = DataHandler(
    np.concatenate((SGN_DATA["gaussian"], BKG_DATA["expo"], REFL_DATA), axis=0),
    var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS, rebin=4
)
DATA_REFL = DataHandler(REFL_DATA, var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)
DATA = DATA.get_binned_data_handler_from_unbinned_data()
DATA_REFL = DATA_REFL.get_binned_data_handler_from_unbinned_data()

FITTER_REFL = F2MassFitter(
    DATA,
    name_signal_pdf=["gaussian"],
    name_refl_pdf=["hist"],
    name_background_pdf=["expo"],
    name="reflection_test",
    chi2_loss=True, tol=1.e-3
)
for key, value in SGN_PARAMS["gaussian"].items():
    FITTER_REFL.set_signal_initpar(0, key, value)
for key, value in BKG_PARAMS["expo"].items():
    FITTER_REFL.set_background_initpar(0, key, value)
FITTER_REFL.set_reflection_template(0, DATA_REFL, 0.5)
FITRES.append(FITTER_REFL.mass_zfit())
FIGS.append(FITTER_REFL.plot_mass_fit(figsize=(10, 10)))
PDFS.append(("gaussian_template", "expo"))


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
    for fitterbinned, pdfs in zip(FITTERS, PDFS):
        signal_pdf_name, _ = pdfs
        rawy, rawy_err = fitterbinned.get_raw_yield()
        assert np.isclose(10000, rawy, atol=5*rawy_err)
        if signal_pdf_name == "gaussian":  # only test bin counting with Gaussian functions for simplicity
            rawy_bc, rawy_bc_err = fitterbinned.get_raw_yield_bincounting()
            assert np.isclose(10000, rawy_bc, atol=5*rawy_bc_err)


def test_plot():
    """
    Test the mass fitter plot
    """
    for fig in FIGS:
        assert isinstance(fig[0], matplotlib.figure.Figure)
        assert isinstance(fig[1], matplotlib.figure.Axes)
