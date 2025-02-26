"""
Simple module with a class to manage the data used in the analysis
"""
import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"  # pylint: disable=wrong-import-position
import numpy as np
import zfit
import pandas as pd
import uproot
from hist import Hist
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
        data: numpy.array / pandas.DataFrame / uproot.behaviors.TH1.Histogram / string
            Data or path to data to be used in the fit
        var_name: str
            Name of the variable used in the fit
        limits: list of floats
            Limits of the x axis used in the fit
        use_zfit: bool
            If True, zfit package is used to fit the data

        **kwargs: dict
            Additional optional arguments:

            - nbins: int
                Number of bins chosen by user to bin data in case of unbinned data

            - rebin: int
                Rebin factor in case of binned data

            - histoname: str
                Name of the histogram to be used in the fit in case of ROOT file

            - treename: str
                Name of the tree to be used in the fit in case of ROOT file
        """
        # Default keyword arguments
        nbins = kwargs.get('nbins', 100)
        rebin = kwargs.get('rebin', 1)

        self._input_ = data
        self._var_name_ = var_name
        self._limits_ = [None, None]
        self._use_zfit_ = use_zfit
        self._obs_ = None
        self._data_ = None
        self._binned_data_ = None
        self._nbins_ = nbins
        self._isbinned_ = None
        self._norm_ = 1.0
        self._rebin_ = rebin
        self.__format__ = None

        if use_zfit:
            data = self.__load_data(data, limits, **kwargs)
            # Update normalization: binned data sums over bin values, unbinned counts entries.
            if self._isbinned_:
                self._binned_data_ = data
                self._norm_ = float(sum(self._binned_data_.values()))
            else:
                self._data_ = data
                self._norm_ = float(len(self._data_.to_pandas()))
        else:
            Logger('Non-zfit data not available', 'FATAL')

    def __check_set_format(self, format_name):
        """
        Checks and sets the data format for the handler.

        If the format is already set and does not match the provided format,
        logs a fatal error. If the format is not set, assigns the provided format.

        Parameters
        ------------------------------------------------
        format_name: str
            The data format to check and set.
        """
        if self.__format__ is not None and self.__format__ != format_name:
            Logger(f'Data format set to {self.__format__}, cannot use {format_name}', 'FATAL')
        elif self.__format__ is None:
            self.__format__ = format_name

    def __check_set_limits_unbinned_obs_(self, data, limits):
        """
        Check and set the limits for unbinned observations.

        This method updates the limits and observation space for unbinned data if the limits are not already set.
        It sets the lower limit to the minimum value in the data and the upper limit to the maximum value in the data.
        The observation space is then updated with these new limits.

        Parameters
        ------------------------------------------------
        data: iterable
            The unbinned data to determine the limits from.
        limits: list
            The limits provided by the user.
        """
        if None in self._limits_ and limits is not None:
            self._limits_[0] = limits[0]
            self._limits_[1] = limits[1]
        elif None in self._limits_:
            self._limits_[0] = min(data)
            self._limits_[1] = max(data)
        if self._obs_ is None:
            self._obs_ = zfit.Space(self._var_name_, lower=self._limits_[0], upper=self._limits_[1])

    def __check_binned_unbinned_(self, isbinned):
        """
        Checks and sets the binning status of the data.

        This method ensures that the binning status of the data is consistent.
        If the binning status has not been set, it sets it to the provided value.
        If the binning status has already been set and the provided value is different,
        it logs a fatal error indicating a data format mismatch.

        Parameters
        ------------------------------------------------
        isbinned: bool
            The binning status to check against the current status.
        """
        if self._isbinned_ is None:
            self._isbinned_ = isbinned
        elif self._isbinned_ is not None and self._isbinned_ != isbinned:
            Logger('Data format mismatch', 'FATAL')

    def __check_set_limits_binned_obs_(self, data, limits):
        """
        Check and set the limits and binning for binned observations.

        This method checks if the limits for the binned observations are set. If not, it sets the number of bins,
        the lower limit, and the upper limit based on the provided data. It then creates a regular binning and
        observation space using zfit. If the limits are already set, it verifies that the bin edges match the
        provided data. If the bin edges do not match, it logs a fatal error.

        Parameters
        ------------------------------------------------
        data: tuple
            A tuple where the second element is an array-like structure containing the bin edges.
        limits: list
            The limits provided by the user.
        """
        if None in self._limits_ and limits is not None:
            idx_min = np.argmin(np.abs(data[1] - limits[0]))
            idx_max = np.argmin(np.abs(data[1] - limits[1]))
            self._limits_[0] = data[1][idx_min]
            self._limits_[1] = data[1][idx_max]
        elif None in self._limits_:
            self._limits_[0] = data[1][0]
            self._limits_[1] = data[1][-1]
        if self._obs_ is None:
            idx_min = np.argmin(np.abs(data[1] - self._limits_[0]))
            idx_max = np.argmin(np.abs(data[1] - self._limits_[1]))
            self._nbins_ = idx_max - idx_min
            binning = zfit.binned.RegularBinning(
                self._nbins_,
                self._limits_[0],
                self._limits_[1],
                name="xaxis"
            )
            self._obs_ = zfit.Space("xaxis", binning=binning)
        else:
            idx_min = np.argmin(np.abs(data[1] - self._limits_[0]))
            idx_max = np.argmin(np.abs(data[1] - self._limits_[1]))
            binning = zfit.binned.RegularBinning(
                self._nbins_,
                self._limits_[0],
                self._limits_[1],
                name="xaxis"
            )
            bin_edges = []
            for i in range(self._nbins_):
                bin_edges.append(binning.bin(i)[0])
            if not np.allclose(bin_edges, data[1][idx_min:idx_max]):
                Logger('Bin edges do not match', 'FATAL')

    def __load_data(self, data, limits, **kwargs):
        """
        Load data from various formats.

        Parameters
        ------------------------------------------------
        data: str, np.ndarray, pd.DataFrame, or uproot.behaviors.TH1.Histogram
            The data to be loaded. It can be a file path (str), a NumPy array,
            a Pandas DataFrame, or an uproot Histogram.
        limits: list
            The limits provided by the user.
        **kwargs:
            Additional keyword arguments to be passed to the specific data loading functions.

        Returns
        -------------------------------------------------
        data: zfit.core.data.Data or zfit.data.BinnedData:
            The loaded data in the appropriate format.
        """
        if isinstance(data, str):
            data = self.__load_from_file(data, limits, **kwargs)
        elif isinstance(data, np.ndarray):
            self.__check_set_format('numpy')
            data = self.__load_from_numpy(data, limits)
        elif isinstance(data, pd.DataFrame):
            self.__check_set_format('pandas')
            data = self.__load_from_pandas(data, limits)
        elif isinstance(data, uproot.behaviors.TH1.Histogram):
            self.__check_set_format('uproot')
            data = self.__load_from_histogram(data, limits)
        else:
            Logger('Data format not supported', 'FATAL')

        return data

    def __load_from_file(self, filename, limits, **kwargs):
        """
        Load data from file-based sources (ROOT or parquet).

        Parameters
        ------------------------------------------------
        filename: str
            The path to the file to be loaded.
        limits: list
            The limits provided by the user.
        **kwargs:
            Additional keyword arguments to be passed to the specific data loading functions.
        """
        if filename.endswith('.root'):
            if self.__format__ is None:
                self.__check_set_format('root')
            if 'histoname' in kwargs:
                with uproot.open(filename, encoding="utf-8") as file:
                    hist = file[kwargs['histoname']]
                return self.__load_from_histogram(hist, limits)
            if 'treename' in kwargs:
                with uproot.open(filename, encoding="utf-8") as file:
                    df = file[kwargs['treename']].arrays(library='pd')
                return self.__load_from_pandas(df, limits)
            Logger('"histoname" not specified. Please specify the name of the histogram to be used', 'FATAL')
            return None
        if filename.endswith('.parquet') or filename.endswith('.parquet.gzip'):
            self.__check_set_format('parquet')
            df = pd.read_parquet(filename)
            return self.__load_from_pandas(df, limits)
        Logger('Data format not supported yet. Please use .root or .parquet', 'FATAL')
        return None

    def __load_from_numpy(self, data, limits):
        """Load a numpy array as unbinned data."""
        self.__check_binned_unbinned_(False)
        self.__check_set_limits_unbinned_obs_(data, limits)
        return zfit.data.Data.from_numpy(obs=self._obs_, array=data)

    def __load_from_pandas(self, df, limits):
        """Load a pandas DataFrame as unbinned data."""
        self.__check_binned_unbinned_(False)
        self.__check_set_limits_unbinned_obs_(df, limits)
        return zfit.data.Data.from_pandas(obs=self._obs_, df=df)

    def __load_from_histogram(self, hist_obj, limits):
        """
        Load an uproot histogram object as binned data.
        """

        self.__check_binned_unbinned_(True)
        hist = hist_obj.to_hist()
        hist = eval(f"hist[::{self._rebin_}j]")  # pylint: disable=eval-used
        hist_array = hist.to_numpy()

        self.__check_set_limits_binned_obs_(hist_array, limits)
        idx_min = np.argmin(np.abs(hist_array[1] - self._limits_[0]))
        idx_max = np.argmin(np.abs(hist_array[1] - self._limits_[1]))

        data = zfit.data.BinnedData.from_tensor(
            self._obs_,
            hist.values()[idx_min:idx_max],
            hist.variances()[idx_min:idx_max]
        )
        return data

    def add_data(self, data, **kwargs):
        """
        Add data to the existing dataset.

        Parameters
        ------------------------------------------------
        data: str, np.ndarray, pd.DataFrame, or uproot.behaviors.TH1.Histogram
            The data to be added.
        **kwargs:
            Additional keyword arguments to be passed to the specific data loading functions.
        """
        if "limits" in kwargs:
            Logger('Limits not needed for adding data', 'FATAL')
        data = self.__load_data(data, limits=None, **kwargs)

        if self._isbinned_:
            self._binned_data_ = zfit.data.concat(
                [self._binned_data_.to_unbinned(), data.to_unbinned()]
            ).to_binned(self._obs_)
            self._norm_ = float(sum(self._binned_data_.values()))
        else:
            self._data_ = zfit.data.concat([self._data_, data])
            self._norm_ = float(len(self._data_.to_pandas()))

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

    def get_binned_obs_from_unbinned_data(self):
        """
        Get the binned obs from unbinned obs

        Returns
        -------------------------------------------------
        binned_obs: zfit.core.space.Space
            The observation space for unbinned data converted to binned data
        """
        bins = self.get_nbins()
        limits = self.get_limits()
        binning = zfit.binned.RegularBinning(bins, limits[0], limits[1], name=self._var_name_)
        obs = zfit.Space(self._var_name_, binning=binning)

        return obs

    def get_unbinned_obs_from_binned_data(self):
        """
        Get the unbinned obs from binned obs

        Returns
        -------------------------------------------------
        unbinned_obs: zfit.core.space.Space
            The observation space for binned data converted to unbinned data
        """
        limits = self.get_limits()
        obs = zfit.Space("xaxis", lower=limits[0], upper=limits[1])

        return obs

    def get_norm(self):
        """
        Get the integral of the data

        Returns
        -------------------------------------------------
        norm: float
            The normalisation value
        """

        return self._norm_

    def get_bin_center(self):
        """
        Get the center of the bins

        Returns
        -------------------------------------------------
        binning: array
            The bin center
        """
        if self.get_is_binned():
            binning = self.get_obs().binning[0]
        else:
            binning = self.get_binned_obs_from_unbinned_data().binning[0]
        bin_center = []
        for bin_ in binning:
            bin_center.append((bin_[0] + bin_[1])/2)
        return bin_center

    def get_nbins(self):
        """
        Get the number of bins

        Returns
        -------------------------------------------------
        nbins: int
            The number of bins
        """
        return self._nbins_

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

    def get_binned_data_from_unbinned_data(self):
        """
        Get the binned data from unbinned data

        Returns
        -------------------------------------------------
        binned_data: float array
            The binned data obtained from unbinned data
        """
        limits = self.get_limits()
        data_np = zfit.run(self.get_data()[self._var_name_])
        data_values, _ = np.histogram(data_np, self.get_nbins(), range=(limits[0], limits[1]))

        return data_values

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

    def to_hist(self, **kwargs):
        """
        returns data in NamedHist

        Parameters
        ------------------------------------------------
        **kwargs: dict
            Additional optional arguments:

            - lower_edge: float
                lower edge (only used in case of originally unbinned data)

            - upper_edge: float
                upper edge (only used in case of originally unbinned data)

            - nbins: int
                number of bins (only used in case of originally unbinned data)

            - axis_title: str
                label of x-axis (only used in case of originally unbinned data)

            - varname: str
                name of variable (needed in case of originally unbinned data)

        Returns
        -------------------------------------------------
        hist: Hist
            The data in a hist.Hist
        """

        if self._isbinned_:
            return self._binned_data_.to_hist()

        if 'varname' not in kwargs:
            Logger('Name of variable needed in case of unbinned data.', 'FATAL')

        varname = kwargs['varname']
        df_unbinned = self._data_.to_pandas()
        data = df_unbinned[varname].to_numpy()

        nbins = kwargs.get('nbins', 100)
        lower_edge = kwargs.get('lower_edge', min(data))
        upper_edge = kwargs.get('upper_edge', max(data))
        axis_title = kwargs.get('axis_title', varname)

        hist = Hist.new.Reg(nbins, lower_edge, upper_edge, name="x", label=axis_title).Double()
        hist.fill(x=data)

        return hist
