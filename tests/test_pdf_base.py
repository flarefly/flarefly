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


def test_defaults_seeded_at_construction():
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")

    assert set(pdf.parameters) == {"mu", "sigma", "frac"}
    assert pdf.get_init_par("sigma") == 0.010
    assert pdf.get_limits_par("sigma") == [0., None]
    assert pdf.get_fix_par("sigma") is False


def test_user_setting_overrides_default():
    pdf = F2PDFBase("expopow", "ExpoPow", "background")

    pdf.set_init_par("lam", 0.)
    pdf.set_fix_par("mass", False)

    assert pdf.get_init_par("lam") == 0.
    assert pdf.get_fix_par("mass") is False  # default is fixed


def test_defaults_chebpol_and_threshold():
    cheb = F2PDFBase("chebpol2", "Chebyshev", "background")
    assert {"c0", "c1", "c2"} <= set(cheb.parameters)

    thr = F2PDFBase("gaussian", "Gaussian", "signal", at_threshold=True)
    assert thr.get_fix_par("massthr") is True
    assert thr.get_fix_par("powerthr") is False


@pytest.mark.parametrize("setter, value", [
    ("set_init_par", 0.01),
    ("set_limits_par", [0., 1.]),
    ("set_fix_par", True),
])
def test_unknown_parameter_fails(setter, value):
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")

    with pytest.raises(RuntimeError, match="sgima"):
        getattr(pdf, setter)("sgima", value)
    assert "sgima" not in pdf.parameters


def test_setitem_shares_and_composes():
    pdf1 = F2PDFBase("gaussian", "Gaussian", "signal")
    pdf2 = F2PDFBase("gaussian", "Gaussian", "signal")

    pdf2["sigma"] = pdf1["sigma"]
    assert pdf2["sigma"] is pdf1["sigma"]

    pdf2["sigma"] = 2 * pdf1["sigma"]
    assert pdf2["sigma"].value == pytest.approx(2 * pdf1.get_init_par("sigma"))


def test_setitem_number_fails():
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")

    with pytest.raises(RuntimeError, match="sigma"):
        pdf["sigma"] = 0.012
    assert pdf.get_init_par("sigma") == 0.010


def test_setitem_unknown_name_fails():
    pdf1 = F2PDFBase("gaussian", "Gaussian", "signal")
    pdf2 = F2PDFBase("gaussian", "Gaussian", "signal")

    with pytest.raises(RuntimeError, match="sgima"):
        pdf2["sgima"] = pdf1["sigma"]
    assert "sgima" not in pdf2.parameters


def test_defaults_not_shared_between_pdfs():
    pdf1 = F2PDFBase("gaussian", "Gaussian", "signal")
    pdf2 = F2PDFBase("gaussian", "Gaussian", "signal")

    pdf1.get_limits_par("sigma")[1] = 1.

    assert pdf2.get_limits_par("sigma") == [0., None]


def test_repr_runs_without_crashing():
    pdf = F2PDFBase("gaussian", "Gaussian", "signal")
    r = repr(pdf)
    assert isinstance(r, str)
