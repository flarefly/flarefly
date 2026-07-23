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


def _import_root():
    """
    Import the ROOT module, needed only by the RooFit backend.

    Returns
    -------------------------------------------------
    ROOT: module
        The ROOT module.
    """
    try:
        import ROOT  # pylint: disable=import-outside-toplevel
        return ROOT
    except ImportError:
        Logger('ROOT not available, cannot use RooFit backend', 'FATAL')
        return None


class ZfitDataHandler:
    """
    Class for managing the data of (ROOT tree, TH1, numpy array, etc.) in zfit format
    """

    def data_range(self, data):
        """Get the range of the data."""
        arr = data.to_numpy()
        return float(arr.min()), float(arr.max())

    def make_unbinned_obs(self, var_name, limits):
        """Make an unbinned observable for zfit."""
        return zfit.Space(obs=var_name, lower=limits[0], upper=limits[1])

    def obs_name(self, obs):
        """Get the variable name carried by the observable."""
        return obs.obs[0]

    def make_binned_obs(self, var_name, limits, nbins):
        """Make a binned observable for zfit."""
        binning = zfit.binned.RegularBinning(
            nbins,
            limits[0],
            limits[1],
            name=var_name
        )
        return zfit.Space(var_name, binning=binning)

    def load_from_pandas(self, obs, df):
        """Load a pandas DataFrame as unbinned data."""
        return zfit.data.Data.from_pandas(obs=obs, df=df)

    def load_from_numpy(self, obs, array):
        """Load a numpy array as unbinned data."""
        return zfit.data.Data.from_numpy(obs=obs, array=array)

    def load_from_zfit_data(self, obs, data):
        """Load a zfit Data object as unbinned data."""
        return data.with_obs(obs)

    def load_from_hist(self, obs, hist, limits):
        """Load from a hist.Hist object as binned data."""
        hist_array = hist.to_numpy()
        idx_min = np.argmin(np.abs(hist_array[1] - limits[0]))
        idx_max = np.argmin(np.abs(hist_array[1] - limits[1]))

        data = zfit.data.BinnedData.from_tensor(
            obs,
            hist.values()[idx_min:idx_max],
            hist.variances()[idx_min:idx_max]
        )
        return data

    def norm(self, data, isbinned):
        """Get the integral of the data."""
        if isbinned:
            return float(sum(data.values()))
        return float(len(data.to_pandas()))

    def add_data(self, data_old, data_new, obs, isbinned):
        """Add data to the existing dataset."""
        if isbinned:
            data = zfit.data.concat(
                [data_old.to_unbinned(), data_new.to_unbinned()]
            ).to_binned(obs)
        else:
            data = zfit.data.concat([data_old, data_new])
        return data, self.norm(data, isbinned)

    def get_binned_obs_from_unbinned_data(self, bins, limits, var_name):
        """
        Get the binned obs from unbinned obs

        Returns
        -------------------------------------------------
        binned_obs: zfit.core.space.Space
            The observation space for unbinned data converted to binned data
        """
        binning = zfit.binned.RegularBinning(bins, limits[0], limits[1], name=var_name)
        obs = zfit.Space(var_name, binning=binning)

        return obs

    def get_unbinned_obs_from_binned_data(self, limits, var_name):
        """
        Get the unbinned obs from binned obs

        Returns
        -------------------------------------------------
        unbinned_obs: zfit.core.space.Space
            The observation space for binned data converted to unbinned data
        """
        obs = zfit.Space(var_name, lower=limits[0], upper=limits[1])

        return obs

    def get_binning(self, obs):
        """Get the binning of the data."""
        return obs.binning[0]

    def to_pandas(self, data):
        """Convert zfit data to pandas DataFrame."""
        return data.to_pandas()

    def to_numpy(self, data):
        """Convert zfit data to numpy array."""
        return data.to_numpy()[:, 0]

    def to_binned(self, data, binned_obs):
        """Convert zfit unbinned data to binned data."""
        return data.to_binned(binned_obs)

    def to_hist(self, data):
        """Convert zfit data to hist.Hist."""

        return data.to_hist()


# pylint: disable=no-member
class RooFitDataHandler:
    """
    Class for managing the data of (ROOT tree, TH1, numpy array, etc.) in RooFit format
    """
    def __init__(self):
        self.root = _import_root()

    def data_range(self, data):
        """Get the range of the data."""
        arr = data.to_numpy()
        return float(arr.min()), float(arr.max())

    def make_unbinned_obs(self, var_name, limits):
        """Make an unbinned observable for RooFit."""
        return self.root.RooRealVar(var_name, var_name, limits[0], limits[1])

    def obs_name(self, obs):
        """Get the variable name carried by the observable."""
        return obs.GetName()

    def make_binned_obs(self, var_name, limits, nbins):
        """Make a binned observable for RooFit."""
        var = self.root.RooRealVar(var_name, var_name, limits[0], limits[1])
        var.setBins(nbins)
        return var

    def load_from_pandas(self, obs, df):
        """Load a pandas DataFrame as unbinned data."""
        return self.root.RooDataSet.from_pandas(df, [obs])

    def load_from_numpy(self, obs, array):
        """Load a numpy array as unbinned data."""
        return self.root.RooDataSet.from_numpy({obs.GetName(): array}, [obs])

    def load_from_zfit_data(self, obs, data):
        """Load a zfit Data object as unbinned data."""
        return self.root.RooDataSet.from_numpy({obs.GetName(): data.to_numpy()}, [obs])

    def load_from_hist(self, obs, hist, limits):
        """Load from a hist.Hist object as binned data."""

        hist_array = hist.to_numpy()
        idx_min = np.argmin(np.abs(hist_array[1] - limits[0]))
        idx_max = np.argmin(np.abs(hist_array[1] - limits[1]))

        counts = hist.values(flow=False)[idx_min:idx_max]
        variances = hist.variances(flow=False)[idx_min:idx_max]

        hist = obs.createHistogram(obs.GetName())

        for i, (c, v) in enumerate(zip(counts, variances)):
            hist.SetBinContent(i + 1, c)
            hist.SetBinError(i + 1, np.sqrt(v))

        return self.root.RooDataHist(obs.GetName(), obs.GetName(), [obs], hist)

    def norm(self, data, isbinned):  # pylint: disable=unused-argument
        """Get the integral of the data."""
        return float(data.sumEntries())

    # pylint: disable=unused-argument
    def add_data(self, data_old, data_new, obs, isbinned):
        """Add data to the existing dataset."""
        if isbinned:
            data_old.add(data_new)
        else:
            data_old.append(data_new)
        return data_old, self.norm(data_old, isbinned)

    def get_binned_obs_from_unbinned_data(self, bins, limits, var_name):
        """
        Get the binned obs from unbinned obs

        Returns
        -------------------------------------------------
        binned_obs: ROOT.RooRealVar
            The variable for unbinned data converted to binned data
        """
        var = self.root.RooRealVar(var_name, var_name, limits[0], limits[1])
        var.setBins(bins)

        return var

    def get_unbinned_obs_from_binned_data(self, limits, var_name):
        """
        Get the unbinned obs from binned obs

        Returns
        -------------------------------------------------
        unbinned_obs: ROOT.RooRealVar
            The variable for binned data converted to unbinned data
        """
        var = self.root.RooRealVar(var_name, var_name, limits[0], limits[1])

        return var

    def get_binning(self, obs):
        """Get the binning of the data."""
        binning = obs.getBinning()
        return [(binning.binLow(i), binning.binHigh(i)) for i in range(binning.numBins())]

    def to_pandas(self, data):
        """Convert RooFit data to pandas DataFrame."""
        return data.to_pandas()

    def to_numpy(self, data):
        """Convert RooFit data to numpy array."""
        # data.get().first() retrieves the variable from the RooFit dataset.
        return data.to_numpy()[data.get().first().GetName()]

    def to_binned(self, data, binned_obs):
        """Convert RooFit unbinned data to binned data."""
        return self.root.RooDataHist("binnedData", "Binned Data", [binned_obs], data)

    def to_hist(self, data):
        """Convert RooFit data to hist.Hist."""
        th1_hist = data.createHistogram(data.get().first().GetName())
        return uproot.from_pyroot(th1_hist).to_hist()


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class DataHandler:
    """
    Class for storing and managing the data of (ROOT tree, TH1, numpy array, etc.)
    """

    def __init__(self, data=None, var_name='xaxis', limits=None, use_zfit=True, **kwargs):
        """
        Initialize the DataHandler class

        Parameters
        ------------------------------------------------
        data: numpy.array / pandas.DataFrame / uproot.behaviors.TH1.Histogram / ROOT.TH1 / string /
              zfit.data.Data / zfit.data.BinnedData
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

        self._backend = self._make_backend(use_zfit)
        self._input = data
        self._var_name = var_name
        self._limits = [None, None]
        self._use_zfit = use_zfit
        self._obs = None
        self._data = None
        self._binned_data = None
        self._nbins = nbins
        self._isbinned = None
        self._norm = 1.0
        self._rebin = rebin
        self._format = None

        data = self._load_data(data, limits, **kwargs)
        if self._isbinned:
            self._binned_data = data
        else:
            self._data = data
        self._norm = self._backend.norm(data, self._isbinned)

    def _make_backend(self, use_zfit):
        """
        Make the backend for the fit

        Parameters
        ------------------------------------------------
        use_zfit: bool
            If True, zfit package is used to fit the data

        Returns
        ------------------------------------------------
        backend: zfit or None
            Backend for the fit
        """
        if use_zfit:
            return ZfitDataHandler()
        return RooFitDataHandler()

    def _check_set_format(self, format_name):
        """
        Checks and sets the data format for the handler.

        If the format is already set and does not match the provided format,
        logs a fatal error. If the format is not set, assigns the provided format.

        Parameters
        ------------------------------------------------
        format_name: str
            The data format to check and set.
        """
        if self._format is not None and self._format != format_name:
            Logger(f'Data format set to {self._format}, cannot use {format_name}', 'FATAL')
        elif self._format is None:
            self._format = format_name

    def _check_binned_unbinned(self, isbinned):
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
        if self._isbinned is None:
            self._isbinned = isbinned
        elif self._isbinned is not None and self._isbinned != isbinned:
            Logger('Data format mismatch', 'FATAL')

    def _data_range(self, data):
        if isinstance(data, pd.DataFrame):
            return min(data[self._var_name]), max(data[self._var_name])
        if isinstance(data, np.ndarray):
            return data.min(), data.max()
        return self._backend.data_range(data)  # zfit

    def _resolve_limits_unbinned(self, data, limits):
        if None not in self._limits:  # Already set limits
            return

        if limits is not None:
            self._limits = [limits[0], limits[1]]
        else:
            self._limits = list(self._data_range(data))

    def _resolve_limits_binned(self, edges, limits):
        if None not in self._limits:
            return
        if limits is not None:
            self._limits = [edges[np.argmin(np.abs(edges - limits[0]))],
                            edges[np.argmin(np.abs(edges - limits[1]))]]
        else:
            self._limits = [edges[0], edges[-1]]

    def _check_binning_matches(self, edges, idx_min, idx_max):
        expected = np.linspace(self._limits[0], self._limits[1], self._nbins + 1)[:-1]
        if not np.allclose(expected, edges[idx_min:idx_max]):
            Logger('Bin edges do not match', 'FATAL')

    def _build_unbinned_obs(self):
        if self._obs is None:
            self._obs = self._backend.make_unbinned_obs(self._var_name, self._limits)

    def _build_binned_obs(self, edges):
        idx_min = np.argmin(np.abs(edges - self._limits[0]))
        idx_max = np.argmin(np.abs(edges - self._limits[1]))
        if self._obs is None:
            self._nbins = int(idx_max - idx_min)
            self._obs = self._backend.make_binned_obs(
                self._var_name, self._limits, self._nbins)
        else:
            self._check_binning_matches(edges, idx_min, idx_max)

    def _load_data(self, data, limits, **kwargs):
        """
        Load data from various formats.

        Parameters
        ------------------------------------------------
        data: str, np.ndarray, pd.DataFrame, uproot.behaviors.TH1.Histogram, or ROOT.TH1
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
            data = self._load_from_file(data, limits, **kwargs)
        elif isinstance(data, np.ndarray):
            self._check_set_format('numpy')
            data = self._load_from_numpy(data, limits)
        elif isinstance(data, pd.DataFrame):
            self._check_set_format('pandas')
            data = self._load_from_pandas(data, limits)
        elif isinstance(data, zfit.data.Data):
            self._check_set_format('zfit_data')
            data = self._load_from_zfit_data(data, limits)
        elif isinstance(data, zfit.data.BinnedData):
            self._check_set_format('zfit_data_binned')
            data = self._load_from_zfit_data_binned(data, limits)
        elif isinstance(data, uproot.behaviors.TH1.Histogram):
            self._check_set_format('uproot')
            data = self._load_from_uproot_histogram(data, limits)
        # Care the position: "TH1" is also in uproot type
        elif "TH1" in str(type(data)):
            self._check_set_format('root_hist')
            tmp = data.Clone("temp_hist")
            tmp.SetDirectory(0)
            data = self._load_from_uproot_histogram(uproot.from_pyroot(tmp), limits)
        elif "RooDataHist" in str(type(data)):
            self._check_set_format('roofit_hist')
            roo_hist = data.Clone("temp_roo_hist")
            hist = data.createHistogram("temp_hist", roo_hist.get(0).first())
            hist.SetDirectory(0)
            data = self._load_from_uproot_histogram(uproot.from_pyroot(hist), limits)
        else:
            Logger(f'Data format {type(data)} not supported', 'FATAL')

        return data

    def _load_from_file(self, filename, limits, **kwargs):
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
            if self._format is None:
                self._check_set_format('root')
            if 'histoname' in kwargs:
                with uproot.open(filename, encoding="utf-8") as file:
                    hist = file[kwargs['histoname']]
                return self._load_from_uproot_histogram(hist, limits)
            if 'treename' in kwargs:
                with uproot.open(filename, encoding="utf-8") as file:
                    df = file[kwargs['treename']].arrays(library='pd')
                return self._load_from_pandas(df, limits)
            Logger('"histoname" not specified. Please specify the name of the histogram to be used', 'FATAL')
            return None
        if filename.endswith('.parquet') or filename.endswith('.parquet.gzip'):
            self._check_set_format('parquet')
            df = pd.read_parquet(filename)
            return self._load_from_pandas(df, limits)
        Logger('Data format not supported yet. Please use .root or .parquet', 'FATAL')
        return None

    def _load_from_numpy(self, data, limits):
        """Load a numpy array as unbinned data."""
        self._check_binned_unbinned(False)
        self._resolve_limits_unbinned(data, limits)
        self._build_unbinned_obs()
        return self._backend.load_from_numpy(obs=self._obs, array=data)

    def _load_from_pandas(self, df, limits):
        """Load a pandas DataFrame as unbinned data."""
        self._check_binned_unbinned(False)
        self._resolve_limits_unbinned(df, limits)
        self._build_unbinned_obs()
        return self._backend.load_from_pandas(obs=self._obs, df=df)

    def _load_from_zfit_data(self, data, limits):
        """Load a zfit Data object as unbinned data."""
        self._check_binned_unbinned(False)
        self._resolve_limits_unbinned(data, limits)
        self._build_unbinned_obs()
        return self._backend.load_from_zfit_data(obs=self._obs, data=data)

    def _load_from_zfit_data_binned(self, data, limits):
        """Load a zfit DataBinned object as binned data."""
        self._check_binned_unbinned(True)
        self._resolve_limits_binned(data.binning[0].edges, limits)
        self._build_binned_obs(data.binning[0].edges)
        return self._backend.load_from_hist(self._obs, data.to_hist(), self._limits)

    def _load_from_uproot_histogram(self, hist_obj, limits):
        """
        Load a histogram object as binned data.
        """
        self._check_binned_unbinned(True)

        hist = hist_obj.to_hist()
        hist = eval(f"hist[::{self._rebin}j]")  # pylint: disable=eval-used
        hist_array = hist.to_numpy()
        self._resolve_limits_binned(hist_array[1], limits)
        self._build_binned_obs(hist_array[1])

        return self._backend.load_from_hist(self._obs, hist, self._limits)

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
        data = self._load_data(data, limits=None, **kwargs)

        data_add, self._norm = self._backend.add_data(
            self._binned_data if self._isbinned else self._data,
            data, obs=self._obs, isbinned=self._isbinned
        )
        if self._isbinned:
            self._binned_data = data_add
        else:
            self._data = data_add

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
            return self._data
        return self._input

    def get_var_name(self):
        """
        Get the variable name

        Returns
        -------------------------------------------------
        var_name: str
            The variable name
        """
        return self._var_name

    def get_limits(self):
        """
        Get the limits of the x axis

        Returns
        -------------------------------------------------
        limits: list
            The range limits of the x axis
        """
        return self._limits

    def get_use_zfit(self):
        """
        Get the use_zfit flag

        Returns
        -------------------------------------------------
        use_zfit: bool
            True if zfit is used to fit the data
        """
        return self._use_zfit

    def get_obs(self):
        """
        Get the observation space

        Returns
        -------------------------------------------------
        obs: zfit.core.space.Space
            The observation space
        """
        return self._obs

    def get_obs_name(self):
        """
        Get the variable name carried by the observation space

        Returns
        -------------------------------------------------
        obs_name: str
            The name of the variable the observation space is defined for
        """
        return self._backend.obs_name(self._obs)

    def get_binned_obs_from_unbinned_data(self):
        """
        Get the binned obs from unbinned obs

        Returns
        -------------------------------------------------
        binned_obs: zfit.core.space.Space
            The observation space for unbinned data converted to binned data
        """
        bins = self._nbins
        limits = self._limits

        return self._backend.get_binned_obs_from_unbinned_data(bins, limits, self._var_name)

    def get_unbinned_obs_from_binned_data(self):
        """
        Get the unbinned obs from binned obs

        Returns
        -------------------------------------------------
        unbinned_obs: zfit.core.space.Space
            The observation space for binned data converted to unbinned data
        """
        limits = self._limits

        return self._backend.get_unbinned_obs_from_binned_data(limits, self._var_name)

    def get_norm(self):
        """
        Get the integral of the data

        Returns
        -------------------------------------------------
        norm: float
            The normalisation value
        """

        return self._norm

    def get_binning(self):
        """
        Get the binning of the data

        Returns
        -------------------------------------------------
        binning: array
            The binning of the data
        """
        if self._isbinned:
            return self._backend.get_binning(self._obs)
        return self._backend.get_binning(self.get_binned_obs_from_unbinned_data())

    def get_bin_center(self):
        """
        Get the center of the bins

        Returns
        -------------------------------------------------
        binning: array
            The bin center
        """
        bin_center = []
        for bin_ in self.get_binning():
            bin_center.append((bin_[0] + bin_[1])/2)
        return bin_center

    def get_bin_edges(self):
        """
        Get the edges of the bins

        Returns
        -------------------------------------------------
        bin_edges: list
            The bin edges
        """
        binning = self.get_binning()
        bin_edges = []
        for bin_ in binning:
            bin_edges.append(bin_[0])
        bin_edges.append(binning[-1][1])
        return bin_edges

    def get_nbins(self):
        """
        Get the number of bins

        Returns
        -------------------------------------------------
        nbins: int
            The number of bins
        """
        return self._nbins

    def get_is_binned(self):
        """
        Get the data type (binned or not)

        Returns
        -------------------------------------------------
        isbinnned: bool
            A flag that indicates if the data is binned
        """
        return self._isbinned

    def get_binned_data(self):
        """
        Get the binned data

        Returns
        -------------------------------------------------
        binned_data: zfit.data.BinnedData
            The binned data
        """
        return self._binned_data

    def get_binned_data_from_unbinned_data(self):
        """
        Get the binned data from unbinned data

        Returns
        -------------------------------------------------
        binned_data: float array
            The binned data obtained from unbinned data
        """
        limits = self._limits
        data_values, _ = np.histogram(self._backend.to_numpy(self._data), self._nbins, range=(limits[0], limits[1]))

        return data_values

    def get_binned_data_handler_from_unbinned_data(self):
        """
        Get a DataHandler with binned data built from unbinned data

        Returns
        -------------------------------------------------
        binned_data_handler: DataHandler
            A DataHandler containing the unbinned data converted to binned data
        """
        return DataHandler(
            self._backend.to_binned(self._data, self.get_binned_obs_from_unbinned_data()),
            var_name=self._var_name,
            limits=self._limits,
            use_zfit=self._use_zfit,
            rebin=1
        )

    def to_pandas(self):
        """
        returns data in pandas df

        Returns
        -------------------------------------------------
        data: pandas.DataFrame
            The data in a pandas DataFrame
        """
        if self._format in ['pandas', 'numpy', 'parquet', 'root', 'zfit_data'] and not self._isbinned:
            return self._backend.to_pandas(self._data)

        Logger('Data format not supported yet for pandas conversion.', 'ERROR')
        return None

    def to_numpy(self):
        """
        returns data in numpy array

        Returns
        -------------------------------------------------
        data: numpy.ndarray
            The data in a numpy array
        """
        if self._format in ['pandas', 'numpy', 'parquet', 'root', 'zfit_data'] and not self._isbinned:
            return self._backend.to_numpy(self._data)

        Logger('Data format not supported yet for numpy conversion.', 'ERROR')
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

        if self._isbinned:
            return self._backend.to_hist(self._binned_data)

        if 'varname' not in kwargs:
            Logger('Name of variable needed in case of unbinned data.', 'FATAL')

        varname = kwargs['varname']
        df_unbinned = self.to_pandas()
        data = df_unbinned[varname].to_numpy()

        nbins = kwargs.get('nbins', 100)
        lower_edge = kwargs.get('lower_edge', min(data))
        upper_edge = kwargs.get('upper_edge', max(data))
        axis_title = kwargs.get('axis_title', varname)

        hist = Hist.new.Reg(nbins, lower_edge, upper_edge, name="x", label=axis_title).Double()
        hist.fill(x=data)

        return hist

    def dump_to_root(self, filename, **kwargs):
        """
        dumps data in ROOT file

        Parameters
        ------------------------------------------------
        filename: str
            The name of the ROOT file to dump the data to

        **kwargs: dict
            Additional optional arguments:
            - option: str
                option (recreate or update)

            - suffix: str
                suffix to append to objects

            - folder: str
                folder in the ROOT file to store the objects
        """

        suffix = kwargs.get('suffix', '')
        option = kwargs.get('option', 'recreate')
        folder = kwargs.get('folder', '')

        if option not in ['recreate', 'update']:
            Logger('Illegal option to save outputs in ROOT file!', 'FATAL')

        open_file = uproot.recreate if option == 'recreate' else uproot.update

        with open_file(filename) as ofile:
            name = '' if folder == '' else folder + '/'
            if self._isbinned:
                hist = self.to_hist()
                name += f"hdata{suffix}"
                ofile[name] = hist
            else:
                tree = self.to_pandas()
                name += f"treedata{suffix}"
                ofile.mktree(name, tree)
