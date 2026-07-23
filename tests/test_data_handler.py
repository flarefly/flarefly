"""Test suite for DataHandler class."""
import os
from hist import Hist
import numpy as np
import pandas as pd
import pytest
import uproot
import zfit

from flarefly import DataHandler

# -------------------------------
# RAW DATA FIXTURES
# -------------------------------
@pytest.fixture
def numpy_data():
    """Load numpy array"""
    return np.array([1.0, 2.0, 3.0, 4.0])

@pytest.fixture
def numpy_data_to_sum():
    """Load a numpy array for data addition tests"""
    return np.array([1.5, 2.5, 3.5])

@pytest.fixture
def pandas_data():
    """Load pandas DataFrame"""
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

@pytest.fixture
def zfit_data(numpy_data):
    """Load zfit Data"""
    return zfit.data.Data.from_numpy(zfit.Space('x', limits=(0, 5)), numpy_data)


@pytest.fixture
def uproot_histogram():
    """Load uproot histogram"""
    return uproot.open(os.path.join(os.getcwd(), "tests/histos_dplus.root"))["hMass_20_40"]


@pytest.fixture
def parquet_file():
    """Path to parquet file"""
    return os.path.join(os.getcwd(), "tests/test_input.parquet")


@pytest.fixture
def root_tree_file():
    """Path to ROOT tree file"""
    return os.path.join(os.getcwd(), "tests/test_input.root")


@pytest.fixture
def root_histo_file():
    """Path to ROOT histogram file"""
    return os.path.join(os.getcwd(), "tests/histos_dplus.root")

@pytest.fixture
def th1_unweighted():
    """Load unweighted TH1 histogram from ROOT file"""
    ROOT = pytest.importorskip("ROOT")
    min_x, max_x = 1.6, 2.1
    nbins = 500
    dx = (max_x - min_x) / nbins
    histo = ROOT.TH1D("histo_unweighted", "histo_unweighted", nbins, min_x, max_x)
    for i in range(1, nbins + 1):
        for _ in range(i):
            histo.Fill(min_x + (i - 0.5) * dx)
    return histo

@pytest.fixture
def th1_weighted():
    """Load weighted TH1 histogram from ROOT file"""
    ROOT = pytest.importorskip("ROOT")
    min_x, max_x = 1.6, 2.1
    nbins = 500
    dx = (max_x - min_x) / nbins
    histo = ROOT.TH1D("histo_weighted", "histo_weighted", nbins, min_x, max_x)
    for i in range(1, nbins + 1):
        histo.Fill(min_x + (i - 0.5) * dx, i)
        histo.SetBinError(i, i/3)  # set some arbitrary error
    return histo

# -------------------------------
# HANDLER FIXTURES - NO LIMITS
# -------------------------------
@pytest.fixture(params=[
    "zfit_numpy", "zfit_pandas", "zfit_zfit", "zfit_parquet", "zfit_root_tree",
    "roofit_numpy", "roofit_pandas", "roofit_zfit", "roofit_parquet", "roofit_root_tree",
    ])
def handler_unbinned_no_limits(request, numpy_data, pandas_data, zfit_data, parquet_file, root_tree_file):
    """
    Unbinned data handlers without explicit limits

    Creates DataHandler instances for various data formats:
    - Numpy array
    - Pandas DataFrame
    - zfit Data
    - Parquet file
    - ROOT TTree file
    """
    configs = {
        "zfit_numpy": (numpy_data, {"var_name": "x"}),
        "zfit_pandas": (pandas_data, {"var_name": "x"}),
        "zfit_zfit": (zfit_data, {"var_name": "x"}),
        "zfit_parquet": (parquet_file, {"var_name": "x"}),
        "zfit_root_tree": (root_tree_file, {"var_name": "x", "treename": "x"}),
        "roofit_numpy": (numpy_data, {"var_name": "x", "use_zfit": False}),
        "roofit_pandas": (pandas_data, {"var_name": "x", "use_zfit": False}),
        "roofit_zfit": (zfit_data, {"var_name": "x", "use_zfit": False}),
        "roofit_parquet": (parquet_file, {"var_name": "x", "use_zfit": False}),
        "roofit_root_tree": (root_tree_file, {"var_name": "x", "treename": "x", "use_zfit": False}),
    }

    if request.param in (
            "roofit_numpy", "roofit_pandas", "roofit_zfit",
            "roofit_parquet", "roofit_root_tree"
    ):
        pytest.importorskip("ROOT")

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


@pytest.fixture(params=[
    "zfit_uproot", "zfit_root_file", "zfit_TH1_unweighted",
    "roofit_uproot", "roofit_root_file", "roofit_TH1_unweighted",
    ])
def handler_binned_unweighted_no_limits(request, uproot_histogram, root_histo_file, th1_unweighted):
    """
    Binned data handlers without explicit limits

    Creates DataHandler instances for various binned data formats:
    - Uproot histogram
    - ROOT histogram file
    - ROOT TH1 histogram 
    """
    configs = {
        "zfit_uproot": (uproot_histogram, {"var_name": "x"}),
        "zfit_root_file": (root_histo_file, {"var_name": "x", "histoname": "hMass_20_40"}),
        "zfit_TH1_unweighted": (th1_unweighted, {"var_name": "x"}),
        "roofit_uproot": (uproot_histogram, {"var_name": "x", "use_zfit": False}),
        "roofit_root_file": (root_histo_file, {"var_name": "x", "histoname": "hMass_20_40", "use_zfit": False}),
        "roofit_TH1_unweighted": (th1_unweighted, {"var_name": "x", "use_zfit": False})
    }

    if request.param in ("roofit_uproot", "roofit_root_file", "roofit_TH1_unweighted"):
        pytest.importorskip("ROOT")

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)

@pytest.fixture(params=["zfit_TH1_weighted", "roofit_TH1_weighted"])
def handler_binned_weighted_no_limits(request, th1_weighted):
    """Binned data handlers without explicit limits"""
    if request.param in {"zfit_TH1_weighted", "roofit_TH1_weighted"}:
        # skip TH1 if ROOT not available
        pytest.importorskip("ROOT")

    configs = {
        "zfit_TH1_weighted": (th1_weighted, {"var_name": "x"}),
        "roofit_TH1_weighted": (th1_weighted, {"var_name": "x", "use_zfit": False}),
    }


    if request.param == "roofit_TH1_weighted":
        pytest.importorskip("ROOT")

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)

# -------------------------------
# HANDLER FIXTURES - WITH LIMITS
# -------------------------------
@pytest.fixture(params=[
    "zfit_numpy", "zfit_pandas", "zfit_zfit", "zfit_parquet", "zfit_root_tree",
    "roofit_numpy", "roofit_pandas", "roofit_zfit", "roofit_parquet", "roofit_root_tree"
    ])
def handler_unbinned_with_limits(request, numpy_data, pandas_data, zfit_data, parquet_file, root_tree_file):
    """Unbinned data handlers with tight limits [1.75, 2.05]"""
    configs = {
        "zfit_numpy": (numpy_data, {"var_name": "x", "limits": [1.75, 2.05]}),
        "zfit_pandas": (pandas_data, {"var_name": "x", "limits": [1.75, 2.05]}),
        "zfit_zfit": (zfit_data, {"var_name": "x", "limits": [1.75, 2.05]}),
        "zfit_parquet": (parquet_file, {"var_name": "x", "limits": [1.75, 2.05]}),
        "zfit_root_tree": (root_tree_file, {"var_name": "x", "treename": "x", "limits": [1.75, 2.05]}),
        "roofit_numpy": (numpy_data, {"var_name": "x", "limits": [1.75, 2.05], "use_zfit": False}),
        "roofit_pandas": (pandas_data, {"var_name": "x", "limits": [1.75, 2.05], "use_zfit": False}),
        "roofit_zfit": (zfit_data, {"var_name": "x", "limits": [1.75, 2.05], "use_zfit": False}),
        "roofit_parquet": (parquet_file, {"var_name": "x", "limits": [1.75, 2.05], "use_zfit": False}),
        "roofit_root_tree": (root_tree_file, {
            "var_name": "x", "treename": "x", "limits": [1.75, 2.05], "use_zfit": False
        })
    }

    if request.param in (
            "roofit_numpy", "roofit_pandas", "roofit_zfit",
            "roofit_parquet", "roofit_root_tree"
    ):
        pytest.importorskip("ROOT")

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


@pytest.fixture(params=[
    "zfit_numpy", "zfit_pandas", "zfit_zfit", "zfit_parquet", "zfit_root_tree",
    "roofit_numpy", "roofit_pandas", "roofit_zfit", "roofit_parquet", "roofit_root_tree"
    ])
def handler_unbinned_larger_limits(request, numpy_data, pandas_data, zfit_data, parquet_file, root_tree_file):
    """Unbinned data handlers with larger limits [-5, 5]"""
    configs = {
        "zfit_numpy": (numpy_data, {"var_name": "x", "limits": [-5, 5]}),
        "zfit_pandas": (pandas_data, {"var_name": "x", "limits": [-5, 5]}),
        "zfit_zfit": (zfit_data, {"var_name": "x", "limits": [-5, 5]}),
        "zfit_parquet": (parquet_file, {"var_name": "x", "limits": [-5, 5]}),
        "zfit_root_tree": (root_tree_file, {"var_name": "x", "treename": "x", "limits": [-5, 5]}),
        "roofit_numpy": (numpy_data, {"var_name": "x", "limits": [-5, 5], "use_zfit": False}),
        "roofit_pandas": (pandas_data, {"var_name": "x", "limits": [-5, 5], "use_zfit": False}),
        "roofit_zfit": (zfit_data, {"var_name": "x", "limits": [-5, 5], "use_zfit": False}),
        "roofit_parquet": (parquet_file, {"var_name": "x", "limits": [-5, 5], "use_zfit": False}),
        "roofit_root_tree": (root_tree_file, {"var_name": "x", "treename": "x", "limits": [-5, 5], "use_zfit": False}),
    }

    if request.param in (
            "roofit_numpy", "roofit_pandas", "roofit_zfit",
            "roofit_parquet", "roofit_root_tree"
    ):
        pytest.importorskip("ROOT")

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


@pytest.fixture(params=[
    "zfit_uproot", "zfit_root_file", "zfit_TH1_unweighted",
    "roofit_uproot", "roofit_root_file", "roofit_TH1_unweighted"
    ])
def handler_binned_unweighted_with_limits(request, uproot_histogram, root_histo_file, th1_unweighted):
    """Binned data handlers with limits [1.75, 2.05]"""
    configs = {
        "zfit_uproot": (uproot_histogram, {"var_name": "x", "limits": [1.75, 2.05], "rebin": 2}),
        "zfit_root_file": (root_histo_file, {"var_name": "x", "histoname": "hMass_20_40", "limits": [1.75, 2.05]}),
        "zfit_TH1_unweighted": (th1_unweighted, {"var_name": "x", "limits": [1.75, 2.05]}),
        "roofit_uproot": (uproot_histogram, {"var_name": "x", "limits": [1.75, 2.05], "rebin": 2, "use_zfit": False}),
        "roofit_root_file": (root_histo_file, {
            "var_name": "x", "histoname": "hMass_20_40", "limits": [1.75, 2.05], "use_zfit": False
        }),
        "roofit_TH1_unweighted": (th1_unweighted, {"var_name": "x", "limits": [1.75, 2.05], "use_zfit": False}),
    }

    if request.param in ("roofit_uproot", "roofit_root_file", "roofit_TH1_unweighted"):
        pytest.importorskip("ROOT")

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)

@pytest.fixture(params=["zfit_TH1_weighted", "roofit_TH1_weighted"])
def handler_binned_weighted_with_limits(request, th1_weighted):
    """Binned data handlers with limits [1.75, 2.05]"""
    configs = {
        "zfit_TH1_weighted": (th1_weighted, {"var_name": "x", "limits": [1.75, 2.05]}),
        "roofit_TH1_weighted": (th1_weighted, {"var_name": "x", "limits": [1.75, 2.05], "use_zfit": False}),
    }

    if request.param == "roofit_TH1_weighted":
        pytest.importorskip("ROOT")

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


# -------------------------------
# BASIC TESTS - UNBINNED
# -------------------------------
def test_unbinned_data_type(handler_unbinned_no_limits):
    """Test that unbinned data returns the type expected by the backend"""
    if handler_unbinned_no_limits.get_use_zfit():
        assert isinstance(handler_unbinned_no_limits.get_data(), zfit.core.data.Data)
    else:
        ROOT = pytest.importorskip("ROOT")
        assert isinstance(handler_unbinned_no_limits.get_data(), ROOT.RooDataSet)
    assert handler_unbinned_no_limits.get_is_binned() is False


def test_unbinned_var_name(handler_unbinned_no_limits):
    """Test that variable name is correctly set"""
    assert handler_unbinned_no_limits.get_var_name() == 'x'


def test_unbinned_obs(handler_unbinned_no_limits):
    """Test that observation space is created"""
    obs = handler_unbinned_no_limits.get_obs()
    if handler_unbinned_no_limits.get_use_zfit():
        assert isinstance(obs, zfit.core.space.Space)
    else:
        ROOT = pytest.importorskip("ROOT")
        assert isinstance(obs, ROOT.RooRealVar)
    assert handler_unbinned_no_limits.get_obs_name() == 'x'


def test_unbinned_norm_positive(handler_unbinned_no_limits):
    """Test that normalization is positive"""
    norm = handler_unbinned_no_limits.get_norm()
    assert norm > 0
    assert isinstance(norm, float)


def test_unbinned_limits_set(handler_unbinned_no_limits):
    """Test that limits are set (either from data or explicit)"""
    limits = handler_unbinned_no_limits.get_limits()
    assert limits[0] is not None
    assert limits[1] is not None

    assert isinstance(limits[0], (int, float))
    assert isinstance(limits[1], (int, float))

    assert limits[0] < limits[1]


# -------------------------------
# BASIC TESTS - BINNED
# -------------------------------
def test_binned_data_type(handler_binned_unweighted_no_limits):
    """Test that binned data returns correct type"""
    if handler_binned_unweighted_no_limits.get_use_zfit():
        assert isinstance(handler_binned_unweighted_no_limits.get_binned_data(), zfit.data.BinnedData)
    else:
        ROOT = pytest.importorskip("ROOT")
        assert isinstance(handler_binned_unweighted_no_limits.get_binned_data(), ROOT.RooDataHist)
    assert handler_binned_unweighted_no_limits.get_is_binned() is True


def test_binned_var_name(handler_binned_unweighted_no_limits):
    """Test that variable name is correctly set for binned data"""
    assert handler_binned_unweighted_no_limits.get_var_name() == 'x'


def test_binned_obs(handler_binned_unweighted_no_limits):
    """Test that binned observation space is created"""
    obs = handler_binned_unweighted_no_limits.get_obs()
    if handler_binned_unweighted_no_limits.get_use_zfit():
        assert isinstance(obs, zfit.core.space.Space)
        assert obs.obs[0] == 'x'
        assert obs.binning is not None
    else:
        ROOT = pytest.importorskip("ROOT")
        assert isinstance(obs, ROOT.RooRealVar)
        assert obs.GetName() == 'x'
        assert obs.getBinning() is not None


def test_binned_norm_positive(handler_binned_unweighted_no_limits):
    """Test that binned normalization is positive"""
    norm = handler_binned_unweighted_no_limits.get_norm()
    assert norm > 0
    assert isinstance(norm, float)


# -------------------------------
# LIMITS TESTS - TIGHT LIMITS
# -------------------------------
def test_unbinned_tight_limits_applied(handler_unbinned_with_limits):
    """Test that tight limits are correctly applied"""
    limits = handler_unbinned_with_limits.get_limits()
    assert np.isclose(limits[0], 1.75)
    assert np.isclose(limits[1], 2.05)


def test_unbinned_tight_limits_reduce_data(handler_unbinned_with_limits, handler_unbinned_no_limits):
    """Test that tight limits reduce the amount of data"""
    norm_with_limits = handler_unbinned_with_limits.get_norm()
    norm_no_limits = handler_unbinned_no_limits.get_norm()
    norm_pandas_with_limits = len(handler_unbinned_with_limits.to_pandas())
    norm_pandas_no_limits = len(handler_unbinned_no_limits.to_pandas())

    # Data with tight limits should have fewer points
    assert norm_with_limits < norm_no_limits
    assert norm_with_limits > 0
    assert norm_pandas_with_limits < norm_pandas_no_limits
    assert norm_pandas_with_limits > 0


def test_unbinned_tight_limits_data_within_range(handler_unbinned_with_limits):
    """Test that all data points are within the specified limits"""
    limits = handler_unbinned_with_limits.get_limits()
    data_values = handler_unbinned_with_limits.to_numpy()

    assert np.all(data_values >= limits[0]), "Some data below lower limit"
    assert np.all(data_values <= limits[1]), "Some data above upper limit"


# -------------------------------
# LIMITS TESTS - LARGER LIMITS
# -------------------------------
def test_unbinned_larger_limits_applied(handler_unbinned_larger_limits):
    """Test that larger limits are correctly set"""
    limits = handler_unbinned_larger_limits.get_limits()
    assert np.isclose(limits[0], -5.0)
    assert np.isclose(limits[1], 5.0)


def test_unbinned_larger_limits_preserve_data(handler_unbinned_larger_limits, handler_unbinned_no_limits):
    """Test that larger limits don't reduce data (all data is within range)"""
    norm_larger = handler_unbinned_larger_limits.get_norm()
    norm_no_limits = handler_unbinned_no_limits.get_norm()

    assert np.isclose(norm_larger, norm_no_limits, rtol=0.01)


def test_unbinned_larger_limits_data_range_unchanged(handler_unbinned_larger_limits):
    """Test that data range is unaffected by larger limits"""
    data_values = handler_unbinned_larger_limits.to_numpy()

    assert data_values.min() > -5.0
    assert data_values.max() < 5.0


# -------------------------------
# LIMITS TESTS - BINNED DATA
# -------------------------------
def test_binned_limits_applied(handler_binned_unweighted_with_limits):
    """Test that limits are correctly applied to binned data"""
    limits = handler_binned_unweighted_with_limits.get_limits()
    assert np.isclose(limits[0], 1.75, atol=0.01)
    assert np.isclose(limits[1], 2.05, atol=0.01)


def test_binned_limits_reduce_bins(handler_binned_unweighted_with_limits, handler_binned_unweighted_no_limits):
    """Test that limits reduce the number of bins"""
    nbins_with_limits = handler_binned_unweighted_with_limits.get_nbins()
    nbins_no_limits = handler_binned_unweighted_no_limits.get_nbins()

    assert nbins_with_limits < nbins_no_limits


# -------------------------------
# DATA ADDITION TESTS
# -------------------------------
def test_add_unbinned_data(numpy_data, numpy_data_to_sum):
    """Test adding unbinned data to existing handler"""
    handler = DataHandler(numpy_data, var_name='x', limits=[-3, 3])
    norm_initial = handler.get_norm()

    handler.add_data(numpy_data_to_sum)
    norm_after = handler.get_norm()

    # Normalization should increase
    assert norm_after > norm_initial
    assert isinstance(handler.get_data(), zfit.core.data.Data)


def test_add_unbinned_data_correct_sum(numpy_data, numpy_data_to_sum):
    """Test that adding data gives correct total normalization"""
    handler1 = DataHandler(numpy_data, var_name='x', limits=[-3, 3])
    handler2 = DataHandler(numpy_data_to_sum, var_name='x', limits=[-3, 3])

    combined = DataHandler(numpy_data, var_name='x', limits=[-3, 3])
    combined.add_data(numpy_data_to_sum)

    expected_norm = handler1.get_norm() + handler2.get_norm()
    assert np.isclose(combined.get_norm(), expected_norm)


def test_add_binned_data(root_histo_file):
    """Test adding binned data to existing handler"""
    handler = DataHandler(
        root_histo_file, histoname="hMass_20_40", limits=[1.75, 2.05]
    )
    norm_initial = handler.get_norm()

    handler.add_data(root_histo_file, histoname="hMass_40_60")
    norm_after = handler.get_norm()

    assert norm_after > norm_initial
    assert isinstance(handler.get_binned_data(), zfit.data.BinnedData)


def test_add_binned_data_correct_sum(root_histo_file):
    """Test that adding binned data gives correct total normalization"""
    handler1 = DataHandler(
        root_histo_file, histoname="hMass_20_40", limits=[1.75, 2.05]
    )
    handler2 = DataHandler(
        root_histo_file, histoname="hMass_40_60", limits=[1.75, 2.05]
    )

    combined = DataHandler(
        root_histo_file, histoname="hMass_20_40", limits=[1.75, 2.05]
    )
    combined.add_data(root_histo_file, histoname="hMass_40_60")

    expected_norm = handler1.get_norm() + handler2.get_norm()
    assert np.isclose(combined.get_norm(), expected_norm)


# -------------------------------
# CONVERSION TESTS
# -------------------------------
def test_to_pandas_conversion(handler_unbinned_no_limits):
    """Test conversion to pandas DataFrame"""
    df = handler_unbinned_no_limits.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == handler_unbinned_no_limits.get_norm()


def test_to_numpy_conversion(handler_unbinned_no_limits):
    """Test conversion to numpy array"""
    arr = handler_unbinned_no_limits.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert len(arr) == handler_unbinned_no_limits.get_norm()


def test_to_hist_conversion_binned(handler_binned_unweighted_no_limits):
    """Test conversion to Hist for binned data"""
    hist = handler_binned_unweighted_no_limits.to_hist(varname='x')
    assert isinstance(hist, Hist)


def test_to_hist_conversion_unbinned(handler_unbinned_no_limits):
    """Test conversion to Hist for unbinned data"""
    hist = handler_unbinned_no_limits.to_hist(varname='x', nbins=50)
    assert isinstance(hist, Hist)


def test_dump_to_root_binned(handler_binned_unweighted_no_limits):
    """Test dumping binned data to ROOT file"""
    output_file = "binned_output.root"
    handler_binned_unweighted_no_limits.dump_to_root(str(output_file))

    with uproot.open(str(output_file)) as f:
        assert "hdata" in f
        data = DataHandler(f["hdata"], var_name='x')

    assert data.get_is_binned() is True
    assert data.get_norm() == handler_binned_unweighted_no_limits.get_norm()
    os.remove(output_file)

def test_dump_to_root_unbinned(handler_unbinned_no_limits):
    """Test dumping unbinned data to ROOT file"""
    output_file = "unbinned_output.root"
    handler_unbinned_no_limits.dump_to_root(str(output_file))

    with uproot.open(str(output_file)) as f:
        assert "treedata" in f
    data = DataHandler(output_file, treename="treedata", var_name='x')

    assert data.get_is_binned() is False
    assert data.get_norm() == handler_unbinned_no_limits.get_norm()
    os.remove(output_file)


# -------------------------------
# BINNING CONVERSION TESTS
# -------------------------------
def test_get_binned_obs_from_unbinned(handler_unbinned_no_limits):
    """Test creating binned observable from unbinned data"""
    handler = handler_unbinned_no_limits
    binned_obs = handler.get_binned_obs_from_unbinned_data()

    if handler.get_use_zfit():
        assert isinstance(binned_obs, zfit.core.space.Space)
        assert binned_obs.binning is not None
    else:
        ROOT = pytest.importorskip("ROOT")
        assert isinstance(binned_obs, ROOT.RooRealVar)

    assert len(handler.get_binning()) == handler.get_nbins()


def test_get_unbinned_obs_from_binned(handler_binned_unweighted_no_limits):
    """Test creating unbinned observable from binned data"""
    handler = handler_binned_unweighted_no_limits
    unbinned_obs = handler.get_unbinned_obs_from_binned_data()

    if handler.get_use_zfit():
        assert isinstance(unbinned_obs, zfit.core.space.Space)
        assert unbinned_obs.binning is None
    else:
        ROOT = pytest.importorskip("ROOT")
        assert isinstance(unbinned_obs, ROOT.RooRealVar)


def test_binned_data_handler_from_unbinned(handler_unbinned_no_limits):
    """Test creating binned DataHandler from unbinned data"""
    binned_handler = handler_unbinned_no_limits.get_binned_data_handler_from_unbinned_data()
    assert isinstance(binned_handler, DataHandler)
    assert binned_handler.get_is_binned() is True


# -------------------------------
# BIN INFO TESTS
# -------------------------------
def test_get_bin_center_binned(handler_binned_unweighted_no_limits):
    """Test getting bin centers for binned data"""
    bin_centers = handler_binned_unweighted_no_limits.get_bin_center()
    assert len(bin_centers) == handler_binned_unweighted_no_limits.get_nbins()
    assert all(isinstance(x, (int, float)) for x in bin_centers)


def test_get_bin_edges_binned(handler_binned_unweighted_no_limits):
    """Test getting bin edges for binned data"""
    bin_edges = handler_binned_unweighted_no_limits.get_bin_edges()
    # Should have n_bins + 1 edges
    assert len(bin_edges) == handler_binned_unweighted_no_limits.get_nbins() + 1
    assert all(isinstance(x, (int, float)) for x in bin_edges)
    # Edges should be monotonically increasing
    assert all(bin_edges[i] < bin_edges[i+1] for i in range(len(bin_edges)-1))


def test_get_bin_center_unbinned(handler_unbinned_no_limits):
    """Test getting bin centers for unbinned data"""
    bin_centers = handler_unbinned_no_limits.get_bin_center()
    assert len(bin_centers) == handler_unbinned_no_limits.get_nbins()


def test_get_binned_data_from_unbinned(handler_unbinned_no_limits):
    """Test binning unbinned data"""
    binned_values = handler_unbinned_no_limits.get_binned_data_from_unbinned_data()
    assert isinstance(binned_values, np.ndarray)
    assert len(binned_values) == handler_unbinned_no_limits.get_nbins()
    assert np.sum(binned_values) <= handler_unbinned_no_limits.get_norm()

# -------------------------------
# HISTOGRAM ERRORS TEST
# -------------------------------
def test_weighted_histos_uncertainties(handler_binned_weighted_no_limits):
    """Test that weighted histograms have correct uncertainties"""
    hist = handler_binned_weighted_no_limits.to_hist()
    values = hist.values()
    variances = hist.variances()

    assert np.all(variances >= 0)
    assert np.allclose(np.sqrt(variances), values/3)

def test_unweighted_histos_uncertainties(handler_binned_unweighted_no_limits):
    """Test that unweighted histograms have correct uncertainties"""
    hist = handler_binned_unweighted_no_limits.to_hist()
    values = hist.values()
    variances = hist.variances()

    assert np.all(variances >= 0)
    assert np.allclose(variances, values)


# -------------------------------
# BACKEND EQUIVALENCE TESTS
# -------------------------------
# Test that both backends (zfit and roofit) produce the same results for the same data

def _both_backends(data, **kwargs):
    """Build the same DataHandler on both backends."""
    pytest.importorskip("ROOT")
    return (DataHandler(data, use_zfit=True, **kwargs),
            DataHandler(data, use_zfit=False, **kwargs))


BINNED_CONFIGS = [
    {},
    {"limits": [1.75, 2.05]},
    {"rebin": 2},
    {"limits": [1.75, 2.05], "rebin": 2},
    {"limits": [1.75, 2.05], "rebin": 5},
]
BINNED_IDS = ["plain", "limits", "rebin2", "limits+rebin2", "limits+rebin5"]


@pytest.mark.parametrize("kwargs", BINNED_CONFIGS, ids=BINNED_IDS)
def test_backends_agree_binned_binning(root_histo_file, kwargs):
    """Both backends must derive the same binning from the same histogram"""
    zf, rf = _both_backends(root_histo_file, histoname="hMass_20_40",
                            var_name="x", **kwargs)

    assert zf.get_nbins() == rf.get_nbins()
    assert np.allclose(zf.get_limits(), rf.get_limits())
    assert np.allclose(zf.get_bin_edges(), rf.get_bin_edges())
    assert np.allclose(zf.get_bin_center(), rf.get_bin_center())


@pytest.mark.parametrize("kwargs", BINNED_CONFIGS, ids=BINNED_IDS)
def test_backends_agree_binned_contents(root_histo_file, kwargs):
    """Both backends must load the same bin contents and uncertainties"""
    zf, rf = _both_backends(root_histo_file, histoname="hMass_20_40",
                            var_name="x", **kwargs)

    assert np.isclose(zf.get_norm(), rf.get_norm())
    assert np.allclose(zf.to_hist().values(), rf.to_hist().values())
    assert np.allclose(zf.to_hist().variances(), rf.to_hist().variances())


@pytest.mark.parametrize("kwargs", BINNED_CONFIGS, ids=BINNED_IDS)
def test_backends_agree_th1_input(th1_unweighted, kwargs):
    """Both backends must agree when fed a ROOT.TH1 directly"""
    zf, rf = _both_backends(th1_unweighted, var_name="x", **kwargs)

    assert zf.get_nbins() == rf.get_nbins()
    assert np.isclose(zf.get_norm(), rf.get_norm())
    assert np.allclose(zf.to_hist().values(), rf.to_hist().values())


def test_backends_agree_weighted_histogram(th1_weighted):
    """A weighted TH1 must keep its uncertainties on both backends"""
    zf, rf = _both_backends(th1_weighted, var_name="x")

    assert np.allclose(zf.to_hist().values(), rf.to_hist().values())
    assert np.allclose(zf.to_hist().variances(), rf.to_hist().variances())
    # the fixture sets error = content / 3, which must survive the round trip
    assert np.allclose(np.sqrt(rf.to_hist().variances()), rf.to_hist().values() / 3)


@pytest.mark.parametrize("kwargs", [{}, {"limits": [1.75, 2.05]}],
                         ids=["plain", "limits"])
def test_backends_agree_unbinned(parquet_file, kwargs):
    """Both backends must load the same unbinned data"""
    zf, rf = _both_backends(parquet_file, var_name="x", **kwargs)

    assert np.isclose(zf.get_norm(), rf.get_norm())
    assert np.allclose(zf.get_limits(), rf.get_limits())
    assert np.allclose(np.sort(zf.to_numpy()), np.sort(rf.to_numpy()))


def test_backends_agree_add_data_binned(root_histo_file):
    """Adding binned data must give the same result on both backends"""
    zf, rf = _both_backends(root_histo_file, histoname="hMass_20_40",
                            var_name="x", limits=[1.75, 2.05])
    zf.add_data(root_histo_file, histoname="hMass_40_60")
    rf.add_data(root_histo_file, histoname="hMass_40_60")

    assert np.isclose(zf.get_norm(), rf.get_norm())


def test_backends_agree_add_data_unbinned(numpy_data, numpy_data_to_sum):
    """Adding unbinned data must give the same result on both backends"""
    zf, rf = _both_backends(numpy_data, var_name="x", limits=[-3, 3])
    zf.add_data(numpy_data_to_sum)
    rf.add_data(numpy_data_to_sum)

    assert np.isclose(zf.get_norm(), rf.get_norm())


# -------------------------------
# ERROR HANDLING TESTS
# -------------------------------

def test_unsupported_data_type_raises():
    """An unsupported input type must be rejected"""
    with pytest.raises(RuntimeError, match="not supported"):
        DataHandler({"not": "data"}, var_name="x")


def test_root_file_without_object_name_raises(root_histo_file):
    """A ROOT file needs either histoname or treename"""
    with pytest.raises(RuntimeError, match="histoname"):
        DataHandler(root_histo_file, var_name="x")


def test_unsupported_file_extension_raises():
    """Only .root and .parquet files are supported"""
    with pytest.raises(RuntimeError, match="not supported yet"):
        DataHandler("some_file.txt", var_name="x")


def test_to_hist_unbinned_without_varname_raises(numpy_data):
    """Converting unbinned data to a histogram needs the variable name"""
    handler = DataHandler(numpy_data, var_name="x")
    with pytest.raises(RuntimeError, match="Name of variable needed"):
        handler.to_hist()


def test_dump_to_root_illegal_option_raises(numpy_data, tmp_path):
    """Only 'recreate' and 'update' are valid dump options"""
    handler = DataHandler(numpy_data, var_name="x")
    with pytest.raises(RuntimeError, match="Illegal option"):
        handler.dump_to_root(str(tmp_path / "out.root"), option="append")


def test_add_data_with_limits_raises(numpy_data, numpy_data_to_sum):
    """Limits cannot be changed while adding data"""
    handler = DataHandler(numpy_data, var_name="x")
    with pytest.raises(RuntimeError, match="Limits not needed"):
        handler.add_data(numpy_data_to_sum, limits=[0, 5])


def test_add_data_format_mismatch_raises(numpy_data, pandas_data):
    """Data added to a handler must have the same format as the original data"""
    handler = DataHandler(numpy_data, var_name="x")
    with pytest.raises(RuntimeError, match="cannot use"):
        handler.add_data(pandas_data)


def test_add_binned_data_to_unbinned_handler_raises(numpy_data, root_histo_file):
    """Binned data cannot be added to an unbinned handler"""
    handler = DataHandler(numpy_data, var_name="x")
    with pytest.raises(RuntimeError, match="mismatch"):
        handler.add_data(root_histo_file, histoname="hMass_20_40")


def test_to_pandas_on_binned_data_returns_none(root_histo_file):
    """Binned data cannot be converted to a DataFrame"""
    handler = DataHandler(root_histo_file, histoname="hMass_20_40", var_name="x")
    assert handler.to_pandas() is None


def test_to_numpy_on_binned_data_returns_none(root_histo_file):
    """Binned data cannot be converted to a numpy array"""
    handler = DataHandler(root_histo_file, histoname="hMass_20_40", var_name="x")
    assert handler.to_numpy() is None


# -------------------------------
# CORRECTNESS TESTS
# -------------------------------

@pytest.fixture
def source_histogram(root_histo_file):
    """The raw histogram contents, read straight from the file with uproot"""
    with uproot.open(root_histo_file) as file:
        hist = file["hMass_20_40"].to_hist()
    return hist.axes[0].edges, hist.values()


@pytest.mark.parametrize("use_zfit", [True, False], ids=["zfit", "roofit"])
def test_binned_contents_match_source(root_histo_file, source_histogram, use_zfit):
    """The loaded bins must be exactly the source bins inside the fit range"""
    if not use_zfit:
        pytest.importorskip("ROOT")
    edges, values = source_histogram
    handler = DataHandler(root_histo_file, histoname="hMass_20_40", var_name="x",
                          limits=[1.75, 2.05], use_zfit=use_zfit)

    idx_min = int(np.argmin(np.abs(edges - 1.75)))
    idx_max = int(np.argmin(np.abs(edges - 2.05)))

    assert handler.get_nbins() == idx_max - idx_min
    assert np.allclose(handler.to_hist().values(), values[idx_min:idx_max])
    assert np.isclose(handler.get_norm(), values[idx_min:idx_max].sum())


@pytest.mark.parametrize("use_zfit", [True, False], ids=["zfit", "roofit"])
def test_no_limits_loads_whole_histogram(root_histo_file, source_histogram, use_zfit):
    """Without limits the whole histogram must be loaded, losing nothing"""
    if not use_zfit:
        pytest.importorskip("ROOT")
    edges, values = source_histogram
    handler = DataHandler(root_histo_file, histoname="hMass_20_40",
                          var_name="x", use_zfit=use_zfit)

    assert handler.get_nbins() == len(values)
    assert np.allclose(handler.get_bin_edges(), edges)
    assert np.isclose(handler.get_norm(), values.sum())


@pytest.mark.parametrize("rebin", [1, 2, 4, 5])
@pytest.mark.parametrize("use_zfit", [True, False], ids=["zfit", "roofit"])
def test_rebin_merges_bins_and_conserves_counts(root_histo_file, source_histogram,
                                                use_zfit, rebin):
    """Rebinning must divide the bin count and preserve the total number of entries"""
    if not use_zfit:
        pytest.importorskip("ROOT")
    _, values = source_histogram
    handler = DataHandler(root_histo_file, histoname="hMass_20_40", var_name="x",
                          rebin=rebin, use_zfit=use_zfit)

    assert handler.get_nbins() == len(values) // rebin
    assert np.isclose(handler.get_norm(), values.sum())
    # each merged bin is the sum of the bins it replaces
    expected = values.reshape(-1, rebin).sum(axis=1)
    assert np.allclose(handler.to_hist().values(), expected)


@pytest.mark.parametrize("use_zfit", [True, False], ids=["zfit", "roofit"])
def test_limits_snap_to_existing_bin_edges(root_histo_file, source_histogram, use_zfit):
    """Requested limits must snap to real bin edges, never split a bin"""
    if not use_zfit:
        pytest.importorskip("ROOT")
    edges, _ = source_histogram
    # limits fall in the middle of a bin
    handler = DataHandler(root_histo_file, histoname="hMass_20_40", var_name="x",
                          limits=[1.7501234, 2.0498765], use_zfit=use_zfit)

    low, high = handler.get_limits()
    assert np.isclose(low, edges[np.argmin(np.abs(edges - 1.7501234))])
    assert np.isclose(high, edges[np.argmin(np.abs(edges - 2.0498765))])
    assert handler.get_nbins() == len(handler.get_bin_edges()) - 1
