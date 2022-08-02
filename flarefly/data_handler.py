"""
Simple module with a class to manage the data used in the analysis
"""

import numpy as np
import zfit
import pandas as pd
import uproot
from flarefly.utils import Logger

# pylint: disable=too-many-instance-attributes
class DataHandler:
    """
    Class for storing and managing the data of (ROOT tree, TH1, numpy array, etc.)
    """

    def __init__(self, data=None, var_name='', limits=None, use_zfit=True, **kwargs):
        """
        Initialize the DataHandler class

        Parameters
        ------------------------------------------------
        data: numpy array
            Data or path to data to be used in the fit
        var_name: str
            Name of the variable used in the fit
        limits: list of floats
            Limits of the x axis used in the fit
        use_zfit: bool
            If True, zfit package is used to fit the data
        **kwargs: dict
            Additional optional arguments:
            - histoname: str
                Name of the histogram to be used in the fit in case of ROOT file
        """
        self._input_ = data
        self._var_name_ = var_name
        self._limits_ = limits if limits is not None else [0, 1]
        self._use_zfit_ = use_zfit
        self._isbinned_ = False

        if use_zfit:
            self._obs_ = zfit.Space(var_name, limits=(limits[0], limits[1]))

            if isinstance(data, str):
                if data.endswith('.root'):
                    self.__format__ = 'root'
                    if 'histoname' in kwargs:
                        histoname = kwargs['histoname']
                        self._obs_ = zfit.Space(histoname, limits=(limits[0], limits[1]))
                        self._isbinned_ = True
                    else:
                        Logger('"histoname" not specified. Please specify the '
                               'name of the histogram to be used', 'FATAL')
                    hist = uproot.open(data)[histoname].to_numpy()
                    self._input_ = hist
                    hist_conv = np.asarray(hist[0], dtype=np.float64)
                    self._data_ = zfit.data.Data.from_numpy(obs=self._obs_, array=hist_conv)
                    del hist_conv, hist
                elif data.endswith('.parquet') or data.endswith('.parquet.gzip'):
                    self.__format__ = 'parquet'
                    self._data_ = pd.read_parquet(data)
                else:
                    Logger('Data format not supported yet. Please use .root or .parquet', 'FATAL')

            elif isinstance(data, np.ndarray):
                self.__format__ = 'ndarray'
                self._data_ = zfit.data.Data.from_numpy(obs=self._obs_, array=data)

            elif isinstance(data, pd.DataFrame):
                self.__format__ = 'pandas'
                self._data_ = zfit.data.Data.from_pandas(obs=self._obs_, df=data)

            else:
                Logger('Data format not supported', 'FATAL')
        else:
            Logger('Non-zfit data not available', 'FATAL')

    def get_data(self, input_data=False):
        """
        Get the data

        Parameters
        ------------------------------------------------
        input_data: bool
            If True, the input data is returned

        Returns
        -------------------------------------------------
        data: zfit.core.data.Data
            The data instance
        """
        if not input_data:
            return self._data_
        return self._input_

    def get_var_name(self):
        """
        Get the variable name

        Returns
        -------------------------------------------------
        var_name: str
            The variable name
        """
        return self._var_name_

    def get_limits(self):
        """
        Get the limits of the x axis

        Returns
        -------------------------------------------------
        limits: list
            The range limits of the x axis
        """
        return self._limits_

    def get_use_zfit(self):
        """
        Get the use_zfit flag

        Returns
        -------------------------------------------------
        limits: list
            The range limits of the x axis
        """
        return self._use_zfit_

    def get_obs(self):
        """
        Get the observation space

        Returns
        -------------------------------------------------
        obs: zfit.core.space.Space
            The observation space
        """
        if self._use_zfit_:
            return self._obs_

        Logger('Observable not available for non-zfit data', 'ERROR')
        return None

    def get_is_binned(self):
        """
        Get the data type (binned or not)

        Returns
        -------------------------------------------------
        isbinnned: bool
            A flag that indicates if the data is binned
        """
        return self._isbinned_

    def to_pandas(self):
        """
        returns data in pandas df

        Returns
        -------------------------------------------------
        data: pandas.DataFrame
            The data in a pandas DataFrame
        """
        if self.__format__ == 'pandas':
            Logger('Data already in pandas format.', 'WARNING')
            return self._input_
        if isinstance(self._input_, np.ndarray):
            self.__format__ = 'pandas'
            return self._data_.to_pandas()

        Logger('Data format not supported yet for pandas conversion.', 'ERROR')
        return None
