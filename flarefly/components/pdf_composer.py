"""
Module containing class for handling the different PDF components
"""
import operator
from functools import reduce
from typing import TYPE_CHECKING
import numpy as np
import zfit
from flarefly.utils import Logger
from flarefly.components import PDFType, F2PDFBase, Parameter, ZfitParameterConverter
from flarefly.pdf_builder import PDFBuilder

if TYPE_CHECKING:
    # No need to import during runtime, only for type checking
    from flarefly.data_handler import DataHandler


# pylint: disable=too-many-instance-attributes
class F2PDFComposer:
    """
    Class used to handle the PDF components for signal and background
    """
    def __init__(
            self,
            data_handler: "DataHandler",
            name_signal_pdf: list[str],
            name_background_pdf: list[str],
            **kwargs
    ):
        """
        Initialize the F2PDFComposer class
        Parameters
        -------------------------------------------------
        data_handler: flarefly.DataHandler
            The data handler containing the data to fit

        name_signal_pdf: list
            The list of names for the signal pdfs. The possible options are:

            - 'nosignal'

            - 'gaussian'

            - 'doublegaus'

            - 'gausexptail'

            - 'genergausexptail'

            - 'genergausexptailsymm'

            - 'bifurgaus'

            - 'crystalball'

            - 'doublecb'

            - 'doublecbsymm'

            - 'genercrystalball'

            - 'cauchy'

            - 'voigtian'

            - 'kde_exact' (requires to set the datasample and options)

            - 'kde_grid' (requires to set the datasample and options)

            - 'kde_fft' (requires to set the datasample and options)

            - 'kde_isj' (requires to set the datasample and options)

            - 'hist' (only for binned fits, requires to set the datasample)

        name_background_pdf: list
            The list of names of the background pdfs. The possible options are:

            - 'nobkg'

            - 'expo'

            - 'powlaw'

            - 'expopow'

            - 'expopowext'

            - 'chebpolN' (N is the order of the polynomial)

            - 'kde_exact' (requires to set the datasample and options)

            - 'kde_grid' (requires to set the datasample and options)

            - 'kde_fft' (requires to set the datasample and options)

            - 'kde_isj' (requires to set the datasample and options)

            - 'hist' (only for binned fits, requires to set the datasample)

        **kwargs: dict
            Additional optional arguments:

            - name_refl_pdf: list
                The list of names of the signal pdfs. It must have the same length as the signal list.
                The possible options are:

                - 'kde_exact' (requires to set the datasample and options)

                - 'kde_grid' (requires to set the datasample and options)

                - 'kde_fft' (requires to set the datasample and options)

                - 'kde_isj' (requires to set the datasample and options)

                - 'hist' (only for binned fits, requires to set the datasample)

            - name: str
                Optional name for the fitter,
                needed in case of multiple fitters defined in the same script

            - extended: bool
                If True, the pdf is considered extended, i.e. the yield is a parameter to fit.
                default value to False

            - limits: list
                list of fit limits to include in the fit

            - signal_at_threshold: list
                list of booleans which indicate whether the signal PDFs are at threshold or not.
                Each element corresponds to a signal pdf

            - label_signal_pdf: list
                list of labels for signal pdfs

            - label_bkg_pdf: list
                list of labels for background pdfs
        """

        self.data_handler = data_handler
        self.is_binned = self.data_handler.get_is_binned()
        self.name = kwargs.get('name', 'fitter')
        self.extended = kwargs.get('extended', False)

        # --- PDF Component Initialization ---
        signal_labels = kwargs.get(
            'label_signal_pdf',
            [f'signal {idx}' for idx in range(len(name_signal_pdf))]
        )
        background_labels = kwargs.get(
            'label_bkg_pdf',
            [f'background {idx}' for idx in range(len(name_background_pdf))]
        )
        signal_at_threshold = kwargs.get('signal_at_threshold', [False for _ in name_signal_pdf])

        self.signal_pdfs = [F2PDFBase(
            name, label, "signal", at_threshold=at_threshold
        ) for name, label, at_threshold in zip(name_signal_pdf, signal_labels, signal_at_threshold)]

        self.background_pdfs = [F2PDFBase(
            name, label, "background"
        ) for name, label in zip(name_background_pdf, background_labels)]

        # --- Logic for empty components ---
        self.no_signal = any(pdf.kind == PDFType.NO_SIGNAL for pdf in self.signal_pdfs)
        self.no_background = any(pdf.kind == PDFType.NO_BKG for pdf in self.background_pdfs)

        if self.no_signal:
            self.signal_pdfs = []
        if self.no_background:
            self.background_pdfs = []

        # --- Reflection Handling (Legacy/Workaround) ---
        refl_names = kwargs.get('name_refl_pdf', ["none" for _ in name_signal_pdf])
        self.refl_pdfs = [F2PDFBase(
            name, f'reflection {idx}', "reflection"
        ) for idx, name in enumerate(refl_names)]

        self.n_refl = 0
        self.refl_idx = [None] * len(self.refl_pdfs)

        if len(self.refl_pdfs) != len(self.signal_pdfs) and not self.no_signal:
            # Logic from original class: check length consistency if not empty
            Logger('List of pdfs for signals and reflections different! Exit', 'FATAL')

        if not all(pdf.kind == PDFType.NONE for pdf in self.refl_pdfs):
            Logger(
                'Reflection pdfs will be deprecated in future versions, '
                'please use background pdfs instead and fix the normalisation '
                "with background_pdfs[i]['frac'] = factor * signal_pdfs[j]['frac']",
                'WARNING'
            )
            n_signal = len(self.signal_pdfs)
            for ipdf, pdf in enumerate(self.refl_pdfs):
                if pdf.kind != PDFType.NONE:
                    self.signal_pdfs.append(pdf)
                    self.refl_idx[ipdf] = n_signal + self.n_refl
                    self.n_refl += 1

        # --- Internal State ---
        self.total_pdf = None
        self.total_pdf_binned = None
        self.total_pdf_norm = None
        self.is_pdf_built = False
        self.converter = None  # flarefly -> zfit parameters, one per build

        # Fractions and Yields containers
        self.fracs = []
        self._init_fracs_storage()
        self.frac_pars = []

        self.total_yield = None
        self.yields = [None for _ in range(len(self.fracs) + 1)]
        self.yield_pars = []
        if self.extended:
            self.total_yield = Parameter(
                f'{self.name}_yield',
                value=self.data_handler.get_norm(),
                limits=[0., None]
            )

        # Limits management
        if 'limits' in kwargs and self.is_binned:
            Logger('Restriction of fit limits is not yet implemented in binned fits!', 'FATAL')

        self.limits = kwargs.get('limits', self.data_handler.get_limits())
        if not isinstance(self.limits[0], list):
            self.limits = [self.limits]
        self.is_truncated = not np.allclose(self.limits, self.data_handler.get_limits())
        self.ratio_truncated = None

        if self.extended and self.is_binned:
            Logger('Binned fit with extended pdf not yet supported!', 'FATAL')

    def build(self):
        """Builds the total PDF and the binned version."""
        self._build_total_pdf()
        self._build_total_pdf_binned()
        self.is_pdf_built = True

    def _init_fracs_storage(self):
        """Initialize the list size for fractions based on fit type."""
        if not self.no_signal and self.no_background:
            if len(self.refl_pdfs) > 0 and self.refl_pdfs[0].kind != PDFType.NONE:
                Logger('Not possible to use reflections without background pdf', 'FATAL')
            self.fracs = [None for _ in range(len(self.signal_pdfs) - 1)]
        elif self.no_signal and not self.no_background:
            self.fracs = [None for _ in range(len(self.background_pdfs) - 1)]
        elif not self.no_signal and not self.no_background:
            self.fracs = [None for _ in range(len(self.signal_pdfs) + len(self.background_pdfs) - 1)]
        else:
            Logger('No signal nor background pdf defined', 'FATAL')

    def _build_signal_pdfs(self, obs: zfit.Space):
        """
        Helper function to compose the signal pdfs
        """
        if self.no_signal:
            Logger('Performing fit with no signal pdf', 'WARNING')
            return

        for ipdf, pdf in enumerate(self.signal_pdfs):
            PDFBuilder.build_pdf(pdf, obs, self.name, ipdf, self.converter, is_signal=True)

    def _build_background_pdfs(self, obs: zfit.Space):
        """
        Helper function to compose the background pdfs
        """
        if self.no_background:
            Logger('Performing fit with no background pdf', 'WARNING')
            return

        for ipdf, pdf in enumerate(self.background_pdfs):
            PDFBuilder.build_pdf(pdf, obs, self.name, ipdf, self.converter, is_signal=False)

            if str(pdf.kind) in ['powlaw', 'expopow', 'expopowext'] and\
                    self.data_handler.get_limits()[0] < pdf.get_init_par("mass"):
                Logger(
                    'The mass parameter in powlaw cannot be smaller than the lower fit limit, '
                    'please fix it.',
                    'FATAL'
                )

    def _pdf_suffix(self, ipdf: int) -> str:
        """
        Helper function to get the parameter name suffix of a pdf from its index in signal + background pdfs
        """
        n_signal = len(self.signal_pdfs)
        return f'signal{ipdf}' if ipdf < n_signal else f'bkg{ipdf - n_signal}'

    def _name_parameters(self):
        """
        Helper function to name the parameters of all the pdfs before converting them,
        a parameter shared between pdfs keeps the name of the first pdf using it
        """
        named = set()
        for ipdf, pdf in enumerate(self.signal_pdfs + self.background_pdfs):
            for par_name, par in pdf.parameters.items():
                if id(par) not in named:
                    par.name = f'{self.name}_{par_name}_{self._pdf_suffix(ipdf)}'
                    named.add(id(par))

    def _get_total_pdf_norm(self):
        """
        Get the normalization of the total pdf
        """
        if self.is_binned:
            self.total_pdf_norm = float(self.data_handler.get_norm())
            return

        norm_total_pdf = 0.
        for lim in self.limits:
            norm_obs = self.data_handler.get_obs().with_limits(lim)
            norm_total_pdf += float(self.data_handler.get_data().with_obs(norm_obs).n_events)
        self.total_pdf_norm = norm_total_pdf

    def _setup_single_pdf(self, f2pdf: F2PDFBase, obs: zfit.Space) -> zfit.pdf.BasePDF:
        """Helper for single component fits."""
        pdf = f2pdf.pdf.copy()
        if self.extended:
            pdf.set_yield(self.converter.convert(self.total_yield))
        if self.is_truncated:
            pdf = pdf.to_truncated(limits=self.limits, obs=obs)
        return pdf

    def _setup_fractions(self):
        """
        Helper function to setup all the fractions
        """
        pdfs = self.signal_pdfs + self.background_pdfs
        self.frac_pars = [pdf['frac'] for pdf in pdfs[:-1]]
        self.fracs = [self.converter.convert(frac) for frac in self.frac_pars]

        frac_last = 1 - reduce(operator.add, self.frac_pars)
        frac_last.name = f'{self.name}_frac_{self._pdf_suffix(len(pdfs) - 1)}'
        self.frac_pars.append(frac_last)

    def _build_total_pdf(self):
        """
        Helper function to compose the total pdf
        """
        if self.is_binned:  # we need unbinned pdfs to sum them
            obs = self.data_handler.get_unbinned_obs_from_binned_data()
        else:
            obs = self.data_handler.get_obs()

        # order of the pdfs is signal, background
        # one converter for all the pdfs, so that shared parameters become a single zfit parameter
        self.converter = ZfitParameterConverter()
        self.frac_pars, self.yield_pars = [], []
        self._name_parameters()
        self._build_signal_pdfs(obs)
        self._build_background_pdfs(obs)

        self._get_total_pdf_norm()

        if self.no_signal and len(self.background_pdfs) == 1:
            self.total_pdf = self._setup_single_pdf(self.background_pdfs[0], obs)
            return

        if self.no_background and len(self.signal_pdfs) == 1:
            self.total_pdf = self._setup_single_pdf(self.signal_pdfs[0], obs)
            return

        self._setup_fractions()

        if self.extended:
            for ipdf, frac in enumerate(self.frac_pars):
                yield_par = frac * self.total_yield
                yield_par.name = f'{self.name}_yield_{self._pdf_suffix(ipdf)}'
                self.yield_pars.append(yield_par)
            self.yields = [self.converter.convert(yield_par) for yield_par in self.yield_pars]

            pdfs_sum = [pdf.pdf.copy() for pdf in self.signal_pdfs + self.background_pdfs]
            for pdf, y in zip(pdfs_sum, self.yields):
                pdf.set_yield(y)
            self.total_pdf = zfit.pdf.SumPDF(pdfs_sum)
        else:
            self.total_pdf = zfit.pdf.SumPDF(
                [pdf.pdf for pdf in self.signal_pdfs+self.background_pdfs],
                self.fracs
            )

        if not self.is_binned and self.is_truncated:
            self.total_pdf = self.total_pdf.to_truncated(limits=self.limits, obs=obs, norm=obs)

    def _build_total_pdf_binned(self):
        """
        Helper function to compose the total pdf binned from unbinned
        """
        # for binned data, obs already contains the wanted binning
        if self.is_binned:
            obs = self.data_handler.get_obs()
        else:
            # for unbinned data,we impose a binning
            obs = self.data_handler.get_binned_obs_from_unbinned_data()

        if self.no_signal and len(self.background_pdfs) == 1:
            self.total_pdf_binned = zfit.pdf.BinnedFromUnbinnedPDF(self.background_pdfs[0].pdf, obs)
            return
        if self.no_background and len(self.signal_pdfs) == 1:
            self.total_pdf_binned = zfit.pdf.BinnedFromUnbinnedPDF(self.signal_pdfs[0].pdf, obs)
            return

        self.total_pdf_binned = zfit.pdf.BinnedFromUnbinnedPDF(zfit.pdf.SumPDF(
            [pdf.pdf for pdf in self.signal_pdfs+self.background_pdfs], self.fracs), obs)

    def _check_consistency_fix_frac(
            self,
            idx_pdf: int,
            target_pdf: int,
            fixed_type: str = 'signal',
            target_type: str = 'signal'
    ):
        """
        Checks the consistency of fixing the fraction between PDFs.

        Parameters
        -------------------------------------------------
        idx_pdf: int
            The index of the PDF to check.
        target_pdf: int
            The target PDF to compare against.
        fixed_type: 'signal' or 'bkg'
            The type of the PDF to check.
        target_type: 'signal' or 'bkg'
            The type of the target PDF to compare against.
        """
        if fixed_type == target_type and idx_pdf == target_pdf:
            Logger(
                f'Index {idx_pdf} is the same as {target_pdf}, '
                'cannot constrain the fraction to itself',
                'FATAL'
            )
        if target_type == 'signal' and target_pdf >= len(self.signal_pdfs):
            Logger(
                f'Target signal index {target_pdf} is out of range',
                'FATAL'
            )
        if target_type == 'bkg' and target_pdf >= len(self.background_pdfs):
            Logger(
                f'Target background index {target_pdf} is out of range',
                'FATAL'
            )

    def add_frac_constraint(
            self,
            idx_pdf: int,
            target_pdf: int,
            factor: float = 1,
            fixed_type: str = 'signal',
            target_type: str = 'signal'
    ):
        """
        Fix the fraction of a PDF to the fraction of another PDF multiplied by a factor.

        Parameters
        -------------------------------------------------
        idx_pdf: int
            Index of the signal fraction to be fixed
        target_pdf: int
            Index of the signal fraction to be used as reference
        factor: float
            Factor to multiply the frac parameter of the target signal
        fixed_type: 'signal' or 'bkg'
            The type of the PDF to fix.
        target_type: 'signal' or 'bkg'
            The type of the target PDF.
        """
        self._check_consistency_fix_frac(idx_pdf, target_pdf, fixed_type, target_type)

        fixed_pdfs = self.signal_pdfs if fixed_type == 'signal' else self.background_pdfs
        target_pdfs = self.signal_pdfs if target_type == 'signal' else self.background_pdfs
        fixed_pdfs[idx_pdf]['frac'] = factor * target_pdfs[target_pdf]['frac']
