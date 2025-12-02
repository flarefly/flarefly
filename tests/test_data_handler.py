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
    return np.array([1.0, 2.0, 3.0, 4.0])

@pytest.fixture
def numpy_data_to_sum():
    return np.array([1.5, 2.5, 3.5])

@pytest.fixture
def pandas_data():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

@pytest.fixture
def zfit_data(numpy_data):
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

# -------------------------------
# HANDLER FIXTURES - NO LIMITS
# -------------------------------
@pytest.fixture(params=["numpy", "pandas", "zfit", "parquet", "root_tree"])
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
        "numpy": (numpy_data, {"var_name": "x"}),
        "pandas": (pandas_data, {"var_name": "x"}),
        "zfit": (zfit_data, {"var_name": "x"}),
        "parquet": (parquet_file, {"var_name": "x"}),
        "root_tree": (root_tree_file, {"var_name": "x", "treename": "x"}),
    }

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


@pytest.fixture(params=["uproot", "root_file"])
def handler_binned_no_limits(request, uproot_histogram, root_histo_file):
    """
    Binned data handlers without explicit limits

    Creates DataHandler instances for various binned data formats:
    - Uproot histogram
    - ROOT histogram file
    """
    configs = {
        "uproot": (uproot_histogram, {"var_name": "x"}),
        "root_file": (root_histo_file, {"var_name": "x", "histoname": "hMass_20_40"}),
    }

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


# -------------------------------
# HANDLER FIXTURES - WITH LIMITS
# -------------------------------
@pytest.fixture(params=["numpy", "pandas", "zfit", "parquet", "root_tree"])
def handler_unbinned_with_limits(request, numpy_data, pandas_data, zfit_data, parquet_file, root_tree_file):
    """Unbinned data handlers with tight limits [1.75, 2.05]"""
    configs = {
        "numpy": (numpy_data, {"var_name": "x", "limits": [1.75, 2.05]}),
        "pandas": (pandas_data, {"var_name": "x", "limits": [1.75, 2.05]}),
        "zfit": (zfit_data, {"var_name": "x", "limits": [1.75, 2.05]}),
        "parquet": (parquet_file, {"var_name": "x", "limits": [1.75, 2.05]}),
        "root_tree": (root_tree_file, {"var_name": "x", "treename": "x", "limits": [1.75, 2.05]}),
    }

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


@pytest.fixture(params=["numpy", "pandas", "zfit", "parquet", "root_tree"])
def handler_unbinned_larger_limits(request, numpy_data, pandas_data, zfit_data, parquet_file, root_tree_file):
    """Unbinned data handlers with larger limits [-5, 5]"""
    configs = {
        "numpy": (numpy_data, {"var_name": "x", "limits": [-5, 5]}),
        "pandas": (pandas_data, {"var_name": "x", "limits": [-5, 5]}),
        "zfit": (zfit_data, {"var_name": "x", "limits": [-5, 5]}),
        "parquet": (parquet_file, {"var_name": "x", "limits": [-5, 5]}),
        "root_tree": (root_tree_file, {"var_name": "x", "treename": "x", "limits": [-5, 5]}),
    }

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


@pytest.fixture(params=["uproot", "root_file"])
def handler_binned_with_limits(request, uproot_histogram, root_histo_file):
    """Binned data handlers with limits [1.75, 2.05]"""
    configs = {
        "uproot": (uproot_histogram, {"var_name": "x", "limits": [1.75, 2.05], "rebin": 2}),
        "root_file": (root_histo_file, {"var_name": "x", "histoname": "hMass_20_40", "limits": [1.75, 2.05]}),
    }

    data, kwargs = configs[request.param]
    return DataHandler(data, **kwargs)


# -------------------------------
# BASIC TESTS - UNBINNED
# -------------------------------
def test_unbinned_data_type(handler_unbinned_no_limits):
    """Test that unbinned data returns correct zfit.Data type"""
    assert isinstance(handler_unbinned_no_limits.get_data(), zfit.core.data.Data)
    assert handler_unbinned_no_limits.get_is_binned() is False


def test_unbinned_var_name(handler_unbinned_no_limits):
    """Test that variable name is correctly set"""
    assert handler_unbinned_no_limits.get_var_name() == 'x'


def test_unbinned_obs(handler_unbinned_no_limits):
    """Test that observation space is created"""
    obs = handler_unbinned_no_limits.get_obs()
    assert isinstance(obs, zfit.core.space.Space)
    assert obs.obs[0] == 'x'


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
def test_binned_data_type(handler_binned_no_limits):
    """Test that binned data returns correct zfit.BinnedData type"""
    assert isinstance(handler_binned_no_limits.get_binned_data(), zfit.data.BinnedData)
    assert handler_binned_no_limits.get_is_binned() is True


def test_binned_var_name(handler_binned_no_limits):
    """Test that variable name is correctly set for binned data"""
    assert handler_binned_no_limits.get_var_name() == 'x'


def test_binned_obs(handler_binned_no_limits):
    """Test that binned observation space is created"""
    obs = handler_binned_no_limits.get_obs()
    assert isinstance(obs, zfit.core.space.Space)
    assert obs.obs[0] == 'x'
    assert obs.binning is not None


def test_binned_norm_positive(handler_binned_no_limits):
    """Test that binned normalization is positive"""
    norm = handler_binned_no_limits.get_norm()
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

    # Data with tight limits should have fewer points
    assert norm_with_limits < norm_no_limits
    assert norm_with_limits > 0


def test_unbinned_tight_limits_data_within_range(handler_unbinned_with_limits):
    """Test that all data points are within the specified limits"""
    limits = handler_unbinned_with_limits.get_limits()
    data_values = handler_unbinned_with_limits.get_data().value()

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
    data_values = handler_unbinned_larger_limits.get_data().value().numpy()

    assert data_values.min() > -5.0
    assert data_values.max() < 5.0


# -------------------------------
# LIMITS TESTS - BINNED DATA
# -------------------------------
def test_binned_limits_applied(handler_binned_with_limits):
    """Test that limits are correctly applied to binned data"""
    limits = handler_binned_with_limits.get_limits()
    assert np.isclose(limits[0], 1.75, atol=0.01)
    assert np.isclose(limits[1], 2.05, atol=0.01)


def test_binned_limits_reduce_bins(handler_binned_with_limits, handler_binned_no_limits):
    """Test that limits reduce the number of bins"""
    nbins_with_limits = handler_binned_with_limits.get_nbins()
    nbins_no_limits = handler_binned_no_limits.get_nbins()

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


def test_to_hist_conversion_binned(handler_binned_no_limits):
    """Test conversion to Hist for binned data"""
    hist = handler_binned_no_limits.to_hist(varname='x')
    assert isinstance(hist, Hist)


def test_to_hist_conversion_unbinned(handler_unbinned_no_limits):
    """Test conversion to Hist for unbinned data"""
    hist = handler_unbinned_no_limits.to_hist(varname='x', nbins=50)
    assert isinstance(hist, Hist)


# -------------------------------
# BINNING CONVERSION TESTS
# -------------------------------
def test_get_binned_obs_from_unbinned(handler_unbinned_no_limits):
    """Test creating binned observable from unbinned data"""
    binned_obs = handler_unbinned_no_limits.get_binned_obs_from_unbinned_data()
    assert isinstance(binned_obs, zfit.core.space.Space)
    assert binned_obs.binning is not None


def test_get_unbinned_obs_from_binned(handler_binned_no_limits):
    """Test creating unbinned observable from binned data"""
    unbinned_obs = handler_binned_no_limits.get_unbinned_obs_from_binned_data()
    assert isinstance(unbinned_obs, zfit.core.space.Space)
    assert unbinned_obs.binning is None


def test_binned_data_handler_from_unbinned(handler_unbinned_no_limits):
    """Test creating binned DataHandler from unbinned data"""
    binned_handler = handler_unbinned_no_limits.get_binned_data_handler_from_unbinned_data()
    assert isinstance(binned_handler, DataHandler)
    assert binned_handler.get_is_binned() is True


# -------------------------------
# BIN INFO TESTS
# -------------------------------
def test_get_bin_center_binned(handler_binned_no_limits):
    """Test getting bin centers for binned data"""
    bin_centers = handler_binned_no_limits.get_bin_center()
    assert len(bin_centers) == handler_binned_no_limits.get_nbins()
    assert all(isinstance(x, (int, float)) for x in bin_centers)


def test_get_bin_edges_binned(handler_binned_no_limits):
    """Test getting bin edges for binned data"""
    bin_edges = handler_binned_no_limits.get_bin_edges()
    # Should have n_bins + 1 edges
    assert len(bin_edges) == handler_binned_no_limits.get_nbins() + 1
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