"""
Simple module with a class to manage the data used in the analysis
"""
import sys
import numpy as np
import zfit
import pandas as pd
import uproot
from flarefly.utils import Logger

class DataHandler:
    """
    Class for storing and managing the data of (ROOT tree, TH1, numpy array, etc.)
    """

    def __init__(self, data = None, var_name = '', limits = [], use_zfit = True, **kwargs):
        """
        Initialize the data handler
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
            Additional arguments
        """
        self._input_ = data
        self._var_name_ = var_name
        self._limits_ = limits
        self._use_zfit_ = use_zfit

        if use_zfit:
            self._obs_ = zfit.Space(f'{var_name}', limits=(limits[0], limits[1]))

            if isinstance(data, str):
                if data.endswith('.root'):
                    self.__format__ = 'root'
                    if 'histoname' in kwargs:
                        histoname = kwargs['histoname']
                        self._obs_ = zfit.Space(f'{histoname}', limits=(limits[0], limits[1]))
                    else:
                        Logger('"histoname" not specified. Please specify the '
                               'name of the histogram to be used', 'FATAL')
                    hist = uproot.open(data)[histoname].to_numpy()
                    self._input_ = hist
                    hist_conv = np.asarray(hist[0], dtype=np.float64)
                    self._data_ = zfit.data.Data.from_numpy(obs=self._obs_, array=hist_conv)
                    del hist_conv, hist
                elif data.endswith('.parquet'):
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
                print('Data format not supported', 'FAIL')
                sys.exit()
        else:
            print('Non-zfit data not available', 'FAIL')
            sys.exit()

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
            Logger('Observable not available for non-zfit data', 'ERROR')
            return None

    def get_data_instance(self):
        """
        Get the data instance
        """
        return self._data_

    def dump_to_pandas(self):
        """
        Dump data in pandas df
        """
        if self.__format__ == 'pandas':
            Logger('Data already in pandas format.', 'WARNING')
        if isinstance(self._input_, np.ndarray):
            self.__format__ = 'pandas'
            return self._data_.to_pandas()
        else:
            Logger('Data format not supported yet for pandas dump.', 'ERROR')
            return None
