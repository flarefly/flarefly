from enum import Enum
from dataclasses import dataclass


class PDFType(Enum):
    """Enumeration of supported PDF types"""

    # Signal PDFs
    NO_SIGNAL = "nosignal"
    GAUSSIAN = "gaussian"
    DOUBLE_GAUS = "doublegaus"
    GAUS_EXP_TAIL = "gausexptail"
    GENER_GAUS_EXP_TAIL = "genergausexptail"
    GENER_GAUS_EXP_TAIL_SYMM = "genergausexptailsymm"
    BIFUR_GAUS = "bifurgaus"
    CRYSTAL_BALL = "crystalball"
    DOUBLE_CB = "doublecb"
    DOUBLE_CB_SYMM = "doublecbsymm"
    GENER_CRYSTAL_BALL = "genercrystalball"
    CAUCHY = "cauchy"
    VOIGTIAN = "voigtian"
    KDE_EXACT = "kde_exact"
    KDE_GRID = "kde_grid"
    KDE_FFT = "kde_fft"
    KDE_ISJ = "kde_isj"
    HIST = "hist"

    # Background PDFs
    NO_BKG = "nobkg"
    CHEBPOL = "chebpol"
    EXPO = "expo"
    POW_LAW = "powlaw"
    EXPO_POW = "expopow"
    EXPO_POW_EXT = "expopowext"


    def is_kde(self) -> bool:
        return self.value.startswith("kde_")

    def is_hist(self) -> bool:
        return self.value == "hist"

    def has_sigma(self) -> bool:
        """Check if PDF type has sigma parameter"""
        return self.value in [
            "gaussian",
            "gausexptail",
            "genergausexptailsymm",
            "crystalball",
            "doublecb",
            "doublecbsymm",
            "voigtian",
            "hist",
        ]

    def has_hwhm(self) -> bool:
        """Check if PDF type has HWHM parameter"""
        return self.value in ["gaussian", "cauchy", "voigtian"]

    def uses_m_not_mu(self) -> bool:
        """Check if PDF uses 'm' instead of 'mu' for mass parameter"""
        return self.value in ["cauchy", "voigtian"]

    def mass_limits(self) -> bool:
        """Check if PDF has mass limits"""
        return self.value in ["powlaw", "expopow", "expopowext"]

@dataclass
class PDFKind:
    kind: PDFType | str
    order: int | None = None

    def __post_init__(self):
        # Allow to use string representation for kind
        if isinstance(self.kind, str):
            self.kind = PDFType(self.kind)

    def __eq__(self, other):
        if isinstance(other, PDFKind):
            return self.kind is other.kind
        if isinstance(other, PDFType):
            return self.kind is other
        return NotImplemented

    def __hash__(self):
        return hash(self.kind)

    def __getattr__(self, name):
        """Delegate unknown attributes/methods to the Enum"""
        return getattr(self.kind, name)

    def __str__(self):
        if self.kind is PDFType.CHEBPOL and self.order is not None:
            return f"{self.kind.value}{self.order}"
        return self.kind.value