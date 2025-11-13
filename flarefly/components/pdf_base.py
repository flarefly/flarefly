from flarefly.utils import Logger
from flarefly.components.pdf_kind import PDFKind, SignalBkgOrRefl

class F2PDFBase:

    def __init__(self, name_pdf, label_pdf, signal_bkg_or_refl, **kwargs):
        if "chebpol" not in name_pdf:
            print("not chebpol")
            self._kind_pdf_ = PDFKind(name_pdf)
        else:
            print("is chebpol!")
            order = int(name_pdf.replace('chebpol', ''))
            self._kind_pdf_ = PDFKind("chebpol", order=order)
        print(self._kind_pdf_)
        self._pdf_ = None
        self._label_pdf_ = label_pdf
        self.set_signal_bkg_or_refl(signal_bkg_or_refl)
        self._hist_sample_ = None
        self._kde_sample_ = None
        self._kde_option_ = None
        self._refl_over_sgn_ = 0.0
        self._at_threshold_ = kwargs.get('at_threshold', False)
        self._pars_ = {}
        self._init_pars_ = {}
        self._limits_pars_ = {}
        self._fix_pars_ = {}
       
    def set_signal_bkg_or_refl(self, signal_bkg_or_refl):
        signal_bkg_or_refl = signal_bkg_or_refl.lower()
        if signal_bkg_or_refl == "signal":
            self._signal_bkg_or_refl_ = SignalBkgOrRefl.SIGNAL
        elif signal_bkg_or_refl == "background":
            self._signal_bkg_or_refl_ = SignalBkgOrRefl.BACKGROUND
        elif signal_bkg_or_refl == "reflection":
            self._signal_bkg_or_refl_ = SignalBkgOrRefl.REFLECTION
        else:
            Logger(f"Invalid value for signal_bkg_or_refl: {signal_bkg_or_refl}"
                   ", expected 'signal', 'background', or 'reflection'.",
                   "FATAL")

    def print(self):
        """Print the PDF information"""
        print(f"PDF: {self._pdf_}, Label: {self._label_pdf_}, Kind: {self._kind_pdf_}, "
              f"Signal/Background/Reflection: {self._signal_bkg_or_refl_.name}, "
              f"At Threshold: {self._at_threshold_}")
        print(f"init_pars: {self._init_pars_}")
        print(f"limits_pars: {self._limits_pars_}")
        print(f"fix_pars: {self._fix_pars_}")
        print(f"Parameters: {self._pars_}")

    @property
    def kind(self):
        """Get the PDF kind"""
        return self._kind_pdf_

    def is_kde(self):
        """Check if this PDF type is KDE"""
        return self._kind_pdf_.is_kde()
    
    def is_hist(self):
        """Check if this PDF type is histogram"""
        return self._kind_pdf_.is_hist()
    
    def uses_m_not_mu(self):
        """Check if this PDF uses m instead of mu"""
        return self._kind_pdf_.uses_m_not_mu()
    
    def has_sigma(self):
        """Check if PDF type has sigma parameter"""
        return self._kind_pdf_.has_sigma()
    
    def has_hwhm(self):
        """Check if PDF type has a HWHM"""
        return self._kind_pdf_.has_hwhm()
    
    @property
    def kde_sample(self):
        """Get the KDE sample"""
        return self._kde_sample_
    
    @property
    def kde_option(self):
        """Get the KDE options"""
        return self._kde_option_
    
    @property
    def hist_sample(self):
        """Get the histogram sample"""
        return self._hist_sample_
    
    @property
    def parameters(self):
        """Get the parameters"""
        return self._pars_
    
    @kde_sample.setter
    def kde_sample(self, value):
        """Set the KDE sample"""
        self._kde_sample_ = value
    
    @kde_option.setter
    def kde_option(self, value):
        """Set the KDE options"""
        self._kde_option_ = value
    
    @hist_sample.setter
    def hist_sample(self, value):
        """Set the histogram sample"""
        self._hist_sample_ = value
    
    @parameters.setter
    def parameters(self, value):
        """Set the parameters"""
        self._pars_ = value
    
    @property
    def at_threshold(self):
        """Get the at_threshold flag"""
        return self._at_threshold_

    def set_pdf(self, pdf):
        """Set the PDF name"""
        self._pdf_ = pdf

    def get_pdf(self):
        """Get the PDF name"""
        return self._pdf_
    
    def get_init_pars(self):
        """Get the parameters dictionary"""
        return self._init_pars_

    def get_limits_pars(self):
        """Get the parameters limits dictionary"""
        return self._limits_pars_

    def get_fix_pars(self):
        """Get the parameters fix dictionary"""
        return self._fix_pars_
    
    def get_init_par(self, name):
        """Get the parameter initial value"""
        return self._init_pars_[name]

    def get_limits_par(self, name):
        """Get the parameter limits"""
        return self._limits_pars_[name]

    def get_fix_par(self, name):
        """Get the parameter fix flag"""
        return self._fix_pars_[name]
    
    def get_label(self):
        """Get the PDF label"""
        return self._label_pdf_
    
    def set_init_pars(self, pars: dict):
        """Set the parameters dictionary"""
        self._init_pars_ = pars

    def set_limits_pars(self, pars: dict):
        """Set the parameters limits dictionary"""
        self._limits_pars_ = pars

    def set_fix_pars(self, pars: dict):
        """Set the parameters fix dictionary"""
        self._fix_pars_ = pars
    
    def set_init_par(self, name, value):
        """Set the parameter"""
        self._init_pars_[name] = value

    def set_limits_par(self, name, value):
        """Set the parameter limits"""
        self._limits_pars_[name] = value

    def set_fix_par(self, name, value):
        """Set the parameter fix flag"""
        self._fix_pars_[name] = value

    def set_default_init_par(self, name, value):
        """Set default parameter"""
        self._init_pars_.setdefault(name, value)

    def set_default_limits_par(self, name, value):
        """Set default parameter limits"""
        self._limits_pars_.setdefault(name, value)

    def set_default_fix_par(self, name, value):
        """Set default parameter fix flag"""
        self._fix_pars_.setdefault(name, value)