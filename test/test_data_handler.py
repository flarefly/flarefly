"""
Test for flarefly.DataHandler
"""

import zfit
import numpy as np
import pandas as pd
from flarefly.data_handler import DataHandler

LIMITS = [1.75, 2.0]
DATASGN = np.random.normal(1.865, 0.010, size=10000)
DATABKG = np.random.uniform(LIMITS[0], LIMITS[1], size=10000)
DATA = DataHandler(np.concatenate((DATASGN, DATABKG), axis=0),
                   var_name=r'$M$ (GeV/$c^{2}$)', limits=LIMITS)

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
