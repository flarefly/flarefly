"""
Test for flarefly.DataHandler
"""

import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"  # pylint: disable=wrong-import-position
import zfit
import numpy as np
import pandas as pd
import uproot
from hist import Hist
from flarefly import DataHandler

DATANP = np.random.normal(0, 1, size=10000)
DATANP_2 = np.random.uniform(-3, 3, size=10000)
DATAPD = pd.DataFrame(pd.DataFrame({'x': DATANP, 'y': DATANP_2}))
DATAUPROOT = uproot.open(os.path.join(os.getcwd(), "tests/histos_dplus.root"))["hMass_20_40"]
DATAZFIT = zfit.data.Data.from_numpy(zfit.Space('x', limits=(-3, 3)), DATANP)

DATAUNBINNED = [
    DataHandler(DATANP, var_name='x', limits=[1.75, 2.0]),
    DataHandler(DATANP, var_name='x'),
    DataHandler(DATAPD, var_name='x', limits=[1.75, 2.0]),
    DataHandler(DATAPD, var_name='x'),
    DataHandler(DATAZFIT, var_name='x'),
    DataHandler(os.path.join(os.getcwd(), "tests/normal_distribution.parquet"), var_name='x', limits=[1.75, 2.0]),
    DataHandler(
        os.path.join(os.getcwd(), "tests/normal_distribution.root"),
        var_name='x', treename="tree", limits=[1.75, 2.0]
    )
]

DATASUMUNBINNED = DataHandler(DATANP, var_name='x', limits=[-3, 3])
DATASUMUNBINNED.add_data(DATANP_2)

DATABINNED = [
    DataHandler(DATAUPROOT, var_name='x', limits=[1.75, 2.0], rebin=2),
    DataHandler(os.path.join(os.getcwd(), "tests/histos_dplus.root"), histoname="hMass_20_40", limits=[1.75, 2.0])
]

DATASUNBINNED = DataHandler(
    os.path.join(os.getcwd(), "tests/histos_dplus.root"),
    histoname="hMass_20_40", limits=[1.75, 2.0]
)
DATASUNBINNED.add_data(os.path.join(os.getcwd(), "tests/histos_dplus.root"), histoname="hMass_40_60")


def test_data():
    """
    Test the get_data() function
    """
    for data in DATAUNBINNED:
        assert isinstance(data.get_data(), zfit.core.data.Data)


def test_data_binned():
    """
    Test the get_data() function
    """
    for data in DATABINNED:
        assert isinstance(data.get_binned_data(), zfit.data.BinnedData)


def test_data_addition():
    """
    Test the get_data() function
    """
    assert isinstance(DATASUMUNBINNED.get_data(), zfit.core.data.Data)
    norm_1 = DataHandler(DATANP, var_name='x', limits=[-3, 3]).get_norm()
    norm_2 = DataHandler(DATANP_2, var_name='x', limits=[-3, 3]).get_norm()
    assert DATASUMUNBINNED.get_norm() == norm_1 + norm_2


def test_data_addition_binned():
    """
    Test the get_data() function
    """
    assert isinstance(DATASUNBINNED.get_binned_data(), zfit.data.BinnedData)
    norm_20_40 = DataHandler(
        os.path.join(os.getcwd(), "tests/histos_dplus.root"), histoname="hMass_20_40", limits=[1.75, 2.0]
    ).get_norm()
    norm_40_60 = DataHandler(
        os.path.join(os.getcwd(), "tests/histos_dplus.root"), histoname="hMass_40_60", limits=[1.75, 2.0]
    ).get_norm()
    assert DATASUNBINNED.get_norm() == norm_20_40 + norm_40_60


def test_obs():
    """
    Test the get_obs() function
    """
    for data in DATAUNBINNED + DATABINNED:
        assert isinstance(data.get_obs(), zfit.core.space.Space)


def test_obs_binned():
    """
    Test the get_obs() function
    """
    for data in DATABINNED:
        assert isinstance(data.get_obs(), zfit.core.space.Space)


def test_pandas_conversion():
    """
    Test the to_pandas() function
    """
    for data in DATAUNBINNED:
        assert isinstance(data.to_pandas(), pd.DataFrame)


def test_histo_conversion():
    """
    Test the to_hist() function
    """
    for data in DATABINNED:
        assert isinstance(data.to_hist(varname='x'), Hist)
