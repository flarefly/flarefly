"""
Test for flarefly.DataHandler
"""

import zfit
import numpy as np
import pandas as pd
from flarefly.data_handler import DataHandler

DATANP = np.random.normal(0, 1, size=10000)
DATA = DataHandler(DATANP, axis=0, var_name='x', limits=[1.75, 2.0])

def test_data():
    """
    Test the get_data() function
    """
    assert isinstance(DATA.get_data(), zfit.core.data.Data)

def test_obs():
    """
    Test the get_obs() function
    """
    assert isinstance(DATA.get_obs(), zfit.core.space.Space)

def test_pandas_conversion():
    """
    Test the to_pandas() function
    """
    assert isinstance(DATA.to_pandas(), pd.DataFrame)
