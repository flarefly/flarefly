"""Module defining the base class for PDFs"""
from flarefly.utils import Logger
from flarefly.components.pdf_kind import PDFKind, SignalBkgOrRefl

class F2PDFBase:  # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Base class for PDFs"""

    def __init__(
            self,
            name_pdf: str,
            label_pdf: str,
            signal_bkg_or_refl: str,
            **kwargs
        ):
        if not name_pdf.startswith("chebpol"):
            print("not chebpol")
            self._kind_pdf = PDFKind(name_pdf)
        else:
            print("is chebpol!")
            order = int(name_pdf.replace('chebpol', ''))
            self._kind_pdf = PDFKind("chebpol", order=order)
        print(self._kind_pdf)
        self._pdf = None
        self._label_pdf = label_pdf
        self.set_signal_bkg_or_refl(signal_bkg_or_refl)
        self._hist_sample = None
        self._kde_sample = None
        self._kde_option = None
        self._refl_over_sgn = 0.0
        self._at_threshold = kwargs.get('at_threshold', False)
        self._pars = {}
        self._parameter_setup = {}

    def set_signal_bkg_or_refl(self, signal_bkg_or_refl):
        """Set whether the PDF is signal, background, or reflection"""
        signal_bkg_or_refl = signal_bkg_or_refl.lower()
        if signal_bkg_or_refl == "signal":
            self._signal_bkg_or_refl = SignalBkgOrRefl.SIGNAL
        elif signal_bkg_or_refl == "background":
            self._signal_bkg_or_refl = SignalBkgOrRefl.BACKGROUND
        elif signal_bkg_or_refl == "reflection":
            self._signal_bkg_or_refl = SignalBkgOrRefl.REFLECTION
        else:
            Logger(f"Invalid value for signal_bkg_or_refl: {signal_bkg_or_refl}"
                   ", expected 'signal', 'background', or 'reflection'.",
                   "FATAL")

    def __repr__(self):
        return (f"F2PDFBase(PDF={self._pdf}, label={self._label_pdf}, kind={self._kind_pdf}, "
            f"Signal/Background/Reflection={self._signal_bkg_or_refl.name}, at_threshold={self._at_threshold}), "
            f"parameter_setup={self._parameter_setup}, parameters={self._pars}")
    
    # ------------------
    # --- Properties ---
    # ------------------
    
    # --- pdf ---
    @property
    def pdf(self):
        """Get the PDF name"""
        return self._pdf

    @pdf.setter
    def pdf(self, value):
        """Set the PDF name"""
        self._pdf = value

    # --- parameters ---
    @property
    def parameters(self):
        """Get the parameters"""
        return self._pars

    @parameters.setter
    def parameters(self, value):
        """Set the parameters"""
        self._pars = value

    # --- kde_sample ---
    @property
    def kde_sample(self):
        """Get the KDE sample"""
        return self._kde_sample

    @kde_sample.setter
    def kde_sample(self, value):
        """Set the KDE sample"""
        self._kde_sample = value

    # --- kde_option ---
    @property
    def kde_option(self):
        """Get the KDE options"""
        return self._kde_option

    @kde_option.setter
    def kde_option(self, value):
        """Set the KDE options"""
        self._kde_option = value

    # --- hist_sample ---
    @property
    def hist_sample(self):
        """Get the histogram sample"""
        return self._hist_sample

    @hist_sample.setter
    def hist_sample(self, value):
        """Set the histogram sample"""
        self._hist_sample = value

    # --- at_threshold ---
    @property
    def at_threshold(self):
        """Get the at_threshold flag"""
        return self._at_threshold
    
    # --- kind ---
    @property
    def kind(self):
        """Get the PDF kind"""
        return self._kind_pdf
    
    # --- label ---
    @property
    def label(self):
        """Get the PDF label"""
        return self._label_pdf

    # ----------------------
    # --- Public Methods ---
    # ----------------------
    def is_kde(self):
        """Check if this PDF type is KDE"""
        return self._kind_pdf.is_kde()

    def is_hist(self):
        """Check if this PDF type is histogram"""
        return self._kind_pdf.is_hist()

    def uses_m_not_mu(self):
        """Check if this PDF uses m instead of mu"""
        return self._kind_pdf.uses_m_not_mu()

    def has_sigma(self):
        """Check if PDF type has sigma parameter"""
        return self._kind_pdf.has_sigma()

    def has_hwhm(self):
        """Check if PDF type has a HWHM"""
        return self._kind_pdf.has_hwhm()
   
    def par_exists(self, name):
        """Check if the parameter exists"""
        return name in self._parameter_setup

    def create_par(self, name):
        """Create a new parameter"""
        self._parameter_setup[name] = {}

    def get_init_par(self, name):
        """Get the parameter initial value"""
        return self._parameter_setup[name]["init"]

    def get_limits_par(self, name):
        """Get the parameter limits"""
        return self._parameter_setup[name]["limits"]

    def get_fix_par(self, name):
        """Get the parameter fix flag"""
        return self._parameter_setup[name]["fix"]

    def get_init_pars(self):
        """Get the parameters initial value dictionary"""
        return {name: p["init"] for name, p in self._parameter_setup.items()}

    def get_limits_pars(self):
        """Get the parameters limits dictionary"""
        return {name: p["limits"] for name, p in self._parameter_setup.items()}

    def get_fix_pars(self):
        """Get the parameters fix dictionary"""
        return {name: p["fix"] for name, p in self._parameter_setup.items()}

    def set_init_par(self, name, value):
        """Set the parameter"""
        self._check_and_create_par(name)
        self._parameter_setup[name]["init"] = value

    def set_limits_par(self, name, value):
        """Set the parameter limits"""
        self._check_and_create_par(name)
        self._parameter_setup[name]["limits"] = value

    def set_fix_par(self, name, value):
        """Set the parameter fix flag"""
        self._check_and_create_par(name)
        self._parameter_setup[name]["fix"] = value

    def set_default_init_par(self, name, value):
        """Set default parameter"""
        self._check_and_create_par(name)
        self._parameter_setup[name].setdefault("init", value)

    def set_default_limits_par(self, name, value):
        """Set default parameter limits"""
        self._check_and_create_par(name)
        self._parameter_setup[name].setdefault("limits", value)

    def set_default_fix_par(self, name, value):
        """Set default parameter fix flag"""
        self._check_and_create_par(name)
        self._parameter_setup[name].setdefault("fix", value)

    def set_default_par(self, name, init=None, limits=None, fix=None):
        """Set default values for a parameter (only if not already defined)."""
        self._check_and_create_par(name)
        if init is not None:
            self._parameter_setup[name].setdefault("init", init)
        if limits is not None:
            self._parameter_setup[name].setdefault("limits", limits)
        if fix is not None:
            self._parameter_setup[name].setdefault("fix", fix)

    # -----------------------
    # --- Private Methods ---
    # -----------------------
    def _check_and_create_par(self, name: str):
        """Check if parameter exists, if not create it"""
        if not self.par_exists(name):
            self.create_par(name)
