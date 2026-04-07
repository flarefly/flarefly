import pytest
from flarefly.components.pdf_base import F2PDFBase
from flarefly.components.pdf_kind import PDFKind, SignalBkgOrRefl


def test_pdf_kind_normal():
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")
    assert isinstance(pdf.kind, PDFKind)
    assert pdf.kind.name == "GAUSSIAN"


def test_pdf_kind_chebpol():
    pdf = F2PDFBase("chebpol3", "Chebyshev", "background")
    assert pdf.kind.name == "CHEBPOL"
    assert pdf.kind.order == 3


@pytest.mark.parametrize("kind_str, enum_val", [
    ("signal", SignalBkgOrRefl.SIGNAL),
    ("background", SignalBkgOrRefl.BACKGROUND),
    ("reflection", SignalBkgOrRefl.REFLECTION)
])
def test_signal_bkg_refl_assignment(kind_str, enum_val):
    pdf = F2PDFBase("gaussian", "test", kind_str)
    assert pdf._signal_bkg_or_refl == enum_val


def test_parameter_creation_and_retrieval():
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")

    pdf.set_init_par("mu", 1.2)
    pdf.set_limits_par("mu", (0, 5))
    pdf.set_fix_par("mu", False)

    assert pdf.get_init_par("mu") == 1.2
    assert pdf.get_limits_par("mu") == (0, 5)
    assert pdf.get_fix_par("mu") is False


def test_parameter_defaults_do_not_override():
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")

    # Set explicit values
    pdf.set_init_par("sigma", 0.3)
    # Now default should NOT overwrite
    pdf.set_default_init_par("sigma", 1.0)

    assert pdf.get_init_par("sigma") == 0.3


def test_default_set_when_empty():
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")

    pdf.set_default_par("alpha", init=0.1, limits=(0, 1), fix=True)

    assert pdf.get_init_par("alpha") == 0.1
    assert pdf.get_limits_par("alpha") == (0, 1)
    assert pdf.get_fix_par("alpha") is True


def test_repr_runs_without_crashing():
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")
    r = repr(pdf)
    assert isinstance(r, str)
