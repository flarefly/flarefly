import os
from hist import Hist
import numpy as np
import pandas as pd
import pytest
import uproot
import zfit

from flarefly.components import F2ComposedPDF
from flarefly import DataHandler

# -----------------
# HANDLER FIXTURES
# -----------------
@pytest.fixture()
def handler_unbinned():
    """
    Unbinned data handlers
    """
    return DataHandler(np.array([1.0, 2.0, 3.0, 4.0]), var_name="x")

@pytest.fixture()
def handler_binned():
    """
    Binned data handlers
    """
    return DataHandler("tests/histos_dplus.root", histoname="hMass_20_40", var_name="x")

def test_init_basic_structure(handler_unbinned):
    """Test standard initialization with one S and one B PDF."""
    model = F2ComposedPDF(handler_unbinned, ['gaussian'], ['expo'], name='test_fitter')

    assert model.name == 'test_fitter'
    assert model.extended is False
    assert len(model.signal_pdfs) == 1
    assert len(model.background_pdfs) == 1
    assert model.no_background is False
    assert model.no_signal is False
    assert len(model.fracs) == 1 # S + B - 1 = 1
    assert model.is_truncated is False
    assert model.is_binned is False
    assert isinstance(model.data_handler, DataHandler)

def test_is_binned_is_set(handler_binned):
    """Test standard initialization with one S and one B PDF."""
    model = F2ComposedPDF(handler_binned, ['gaussian'], ['expo'], name='test_fitter')

    assert model.is_binned is True

@pytest.mark.parametrize("signal_name, background_name, no_signal, no_background", [
    (["gaussian"], ["expo"], False, False), # 1S + 1B
    (["gaussian", "gaussian"], ["expo"], False, False), # 2S + 1B
    (["gaussian"], ["expo", "expo"], False, False), # 1S + 2B
    (["gaussian", "gaussian"], ["expo", "expo"], False, False), # 2S + 2B
    (["gaussian"], ["nobkg"], False, True), # S only
    (["nosignal"], ["expo"], True, False), # B only
])
def test_no_signal_no_background(signal_name, background_name, no_signal, no_background, handler_unbinned):
    """Test no_signal and no_background flags."""
    model = F2ComposedPDF(handler_unbinned, signal_name, background_name, name='test_fitter')

    assert model.no_signal is no_signal
    assert model.no_background is no_background

def test_raises_with_no_pdfs(handler_unbinned):
    """Test that an error is raised when no signal and no background PDFs are provided."""
    with pytest.raises(RuntimeError, match="No signal nor background pdf defined"):
        F2ComposedPDF(handler_unbinned, ['nosignal'], ['nobkg'], name='test_fitter')

def test_init_reflection_handling(handler_unbinned):
    """Test reflection PDFs are correctly appended to signal_pdfs and refl_idx is set."""
    model = F2ComposedPDF(
        handler_unbinned,
        ['gaussian', 'gaussian'],
        ['expo'],
        name_refl_pdf=['none', 'kde_exact']
    )
    assert len(model.signal_pdfs) == 3

    assert model.refl_idx == [None, 2]
    assert model.n_refl == 1

def test_init_reflection_inconsistency_raises(handler_unbinned):
    """Test FATAL error when refl_pdfs length does not match signal_pdfs length."""
    # signal_pdf list size is 1, refl_pdf list size is 2
    with pytest.raises(RuntimeError, match='List of pdfs for signals and reflections different! Exit'):
        F2ComposedPDF(
            handler_unbinned,
            ['gaussian'],
            ['expo'],
            name_refl_pdf=['none', 'kde_exact']
        )

def test_add_frac_constraint_registration(handler_unbinned):
    """Test add_frac_constraint registers the constraint info."""
    model = F2ComposedPDF(handler_unbinned, ['gaussian', 'gaussian'], ['expo', 'expo'])

    # Constraint S1 (idx 1) to B0 (target 0) with factor 0.5
    model.add_frac_constraint(
        idx_pdf=1, target_pdf=0, factor=0.5, fixed_type='signal', target_type='bkg'
    )

    assert len(model.fix_fracs_to_pdfs) == 1
    constraint = model.fix_fracs_to_pdfs[0]

    assert constraint['fixed_pdf_idx'] == 1
    assert constraint['target_pdf_idx'] == 0
    assert constraint['fixed_pdf_type'] == 'signal'
    assert constraint['target_pdf_type'] == 'bkg'
    assert isinstance(constraint['factor'], zfit.Parameter)
    assert constraint['factor'].value().numpy() == 0.5

def test_add_frac_constraint_invalid_index_fatal(handler_unbinned):
    """Test invalid index checks (e.g., target index out of range)."""
    model = F2ComposedPDF(handler_unbinned, ['gaussian'], ['expo']) # S list size 1, B list size 1

    with pytest.raises(RuntimeError, match='Target signal index 1 is out of range'):
        model.add_frac_constraint(0, 1, target_type='signal')

    with pytest.raises(RuntimeError, match='Target background index 1 is out of range'):
        model.add_frac_constraint(0, 1, target_type='bkg')

    with pytest.raises(RuntimeError, match='Index 0 is the same as 0, cannot constrain the fraction to itself'):
        model.add_frac_constraint(0, 0)

def test_build_model_basic(handler_unbinned):
    """Test building the model with one S and one B PDF."""
    model = F2ComposedPDF(handler_unbinned, ['gaussian'], ['expo'], name='test_fitter')
    model.build()

    assert model.is_pdf_built is True
    assert isinstance(model.signal_pdfs[0].pdf, zfit.pdf.BasePDF)
    assert isinstance(model.background_pdfs[0].pdf, zfit.pdf.BasePDF)
    assert isinstance(model.total_pdf, zfit.pdf.SumPDF)
    assert isinstance(model.total_pdf_binned, zfit.pdf.BinnedFromUnbinnedPDF)

def test_setup_fractions_creates_composed_parameter(handler_unbinned):
    """Test that _setup_fractions correctly turns a constrained frac into a ComposedParameter."""
    model = F2ComposedPDF(handler_unbinned, ['gaussian', 'gaussian'], ['expo', 'expo']) # Fracs size 3: [s0, s1, b0]

    model.add_frac_constraint(
        idx_pdf=0, target_pdf=0, factor=2.0, fixed_type='signal', target_type='bkg'
    )

    model._setup_fractions()

    assert isinstance(model.fracs[2], zfit.Parameter)
    assert model.fracs[2].name == 'fitter_frac_bkg0'

    assert isinstance(model.fracs[0], zfit.ComposedParameter)

    target_par = model.fracs[2]
    factor_par = model.fix_fracs_to_pdfs[0]['factor']

    assert model.fracs[0].get_params(floating=None) == {target_par, factor_par}

    expected_value = target_par.value().numpy() * 2.0
    assert model.fracs[0].value().numpy() == expected_value
