"""
Simple module with a class to manage the data used in the analysis
"""
from logging import root
import matplotlib.pyplot as plt
import numpy as np
import zfit
import pandas as pd

class DataHandler:
    """
    Class for storing and managing the data of (ROOT tree, TH1, numpy array, etc.)
    """

    def __init__(self, data = None, var_name = '', limits = [], use_zfit = True):
        """
        Initialize the data handler
        Parameters
        ------------------------------------------------
        data: numpy array
            Data to be used in the fit
        var_name: str
            Name of the variable used in the fit
        limits: list of floats
            Limits of the x axis used in the fit
        use_zfit: bool
            If True, zfit package is used to fit the data
        """
        self._input_ = data
        self._var_name_ = var_name
        self._limits_ = limits
        self._use_zfit_ = use_zfit

        if use_zfit:
            self._obs_ = zfit.Space(f'{var_name}', limits=(limits[0], limits[1]))
            if isinstance(data, np.ndarray):
                self._data_ = zfit.data.Data.from_numpy(obs=self._obs_, array=data)
            if isinstance(data, pd.DataFrame):
                self._data_ = zfit.data.Data.from_pandas(obs=self._obs_, df=data)

    def get_data(self, input_data = False):
        """
        Get the data
        input_data: bool
            If True, the input data is returned
        """
        if not input_data:
            return self._data_
        else:
            return self._input_

    def get_var_name(self):
        """
        Get the variable name
        """
        return self._var_name_

    def get_limits(self):
        """
        Get the limits of the x axis
        """
        return self._limits_

    def get_use_zfit(self):
        """
        Get the use_zfit flag
        """
        return self._use_zfit_

    def get_obs(self):
        """
        Get the observation space
        """
        if self._use_zfit_:
            return self._obs_
        else:
            print('Observable not available for non-zfit data')
            return None

    def get_data_instance(self):
        """
        Get the data instance
        """
        return self._data_

    def get_data_to_pandas(self):
        """
        Get the data in pandas format
        """
        if isinstance(self._input_, pd.DataFrame):
            return self._input_
        if isinstance(self._input_, np.ndarray):
            return self._data_.to_pandas()
        else:
            print('TODO: Data not available in pandas format')
            return None
