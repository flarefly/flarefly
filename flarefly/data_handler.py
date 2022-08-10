"""
Simple module with a class to manage the data used in the analysis
"""

import numpy as np
import zfit
import pandas as pd
import uproot
from flarefly.utils import Logger

# pylint: disable=too-many-instance-attributes,too-many-statements,too-many-branches
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
        self._limits_ = limits if limits is not None else [-1, -1]
        self._use_zfit_ = use_zfit
        self._obs_ = None
        self._data_ = None
        self._binned_data_ = None
        self._isbinned_ = False
        self._norm_ = 1.

        if use_zfit:
            if isinstance(data, str):
                if data.endswith('.root'):
                    self.__format__ = 'root'
                    if 'histoname' in kwargs:
                        hist = uproot.open(data)[kwargs['histoname']]
                        hist_array = hist.to_numpy()
                        if limits is None:
                            self._binned_data_ = zfit.data.BinnedData.from_hist(hist.to_hist())
                            nbins = len(hist_array[1]) - 1
                            self._limits_[0] = hist_array[1][0]
                            self._limits_[1] = hist_array[1][-1]
                        else:
                            idx_min = np.argmin(np.abs(hist_array[1]-self._limits_[0]))
                            idx_max = np.argmin(np.abs(hist_array[1]-self._limits_[1]))
                            nbins = len(hist_array[1][idx_min:idx_max+1])
                            self._limits_[0] = hist_array[1][idx_min]
                            self._limits_[1] = hist_array[1][idx_max]
                        binning = zfit.binned.RegularBinning(
                            nbins,
                            self._limits_[0],
                            self._limits_[1],
                            name="xaxis"
                        )
                        self._obs_ = zfit.Space("xaxis", binning=binning)
                        self._binned_data_ = zfit.data.BinnedData.from_tensor(
                            self._obs_, hist_array[0][idx_min:idx_max+1],
                            np.sqrt(hist_array[0][idx_min:idx_max+1]))
                        self._isbinned_ = True
                    elif 'treename' in kwargs:
                        input_df = uproot.open(data)[kwargs['treename']].arrays(library='pd')
                        if limits is None:
                            self._limits_[0] = min(input_df[self._var_name_])
                            self._limits_[1] = max(input_df[self._var_name_])
                        self._obs_ = zfit.Space(self._var_name_, limits=(self._limits_[0], self._limits_[1]))
                        self._data_ = zfit.data.Data.from_pandas(obs=self._obs_, df=input_df)
                    else:
                        Logger('"histoname" not specified. Please specify the '
                               'name of the histogram to be used', 'FATAL')
                elif data.endswith('.parquet') or data.endswith('.parquet.gzip'):
                    self.__format__ = 'parquet'
                    input_df = pd.read_parquet(data)
                    if limits is None:
                        self._limits_[0] = min(input_df[self._var_name_])
                        self._limits_[1] = max(input_df[self._var_name_])
                    self._obs_ = zfit.Space(self._var_name_, limits=(self._limits_[0], self._limits_[1]))
                    self._data_ = zfit.data.Data.from_pandas(obs=self._obs_, df=input_df)
                else:
                    Logger('Data format not supported yet. Please use .root or .parquet', 'FATAL')

            elif isinstance(data, np.ndarray):
                self.__format__ = 'numpy'
                if limits is None:
                    self._limits_[0] = min(data)
                    self._limits_[1] = max(data)
                self._obs_ = zfit.Space(self._var_name_, limits=(self._limits_[0], self._limits_[1]))
                self._data_ = zfit.data.Data.from_numpy(obs=self._obs_, array=data)

            elif isinstance(data, pd.DataFrame):
                self.__format__ = 'pandas'
                if limits is None:
                    self._limits_[0] = min(data[self._var_name_])
                    self._limits_[1] = max(data[self._var_name_])
                self._obs_ = zfit.Space(self._var_name_, limits=(self._limits_[0], self._limits_[1]))
                self._data_ = zfit.data.Data.from_pandas(obs=self._obs_, df=data)

            else:
                Logger('Data format not supported', 'FATAL')

            if self._isbinned_:
                self._norm_ = float(sum(self._binned_data_.values()))
            else:
                self._norm_ = float(len(self._data_.to_pandas()))
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

    def get_norm(self):
        """
        Get the integral of the data

        Returns
        -------------------------------------------------
        norm: float
            The normalisation value
        """

        return self._norm_

    def get_is_binned(self):
        """
        Get the data type (binned or not)

        Returns
        -------------------------------------------------
        isbinnned: bool
            A flag that indicates if the data is binned
        """
        return self._isbinned_

    def get_binned_data(self):
        """
        Get the binned data

        Returns
        -------------------------------------------------
        binned_data: zfit.data.BinnedData
            The binned data
        """
        return self._binned_data_

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
        if self.__format__ in ['numpy', 'parquet', 'root'] and not self._isbinned_:
            return self._data_.to_pandas()

        Logger('Data format not supported yet for pandas conversion.', 'ERROR')
        return None
