"""Test suite for parameters class."""
import os
from hist import Hist
import numpy as np
import pandas as pd
import pytest
import uproot
import zfit

from flarefly.components import Parameter, ParKind


# -------------------------------
# PARAMETER FIXTURES
# -------------------------------
@pytest.fixture
def parameters():
    p1 = Parameter("p1", 1.0, limits=(0, 5), floating=True)
    p2 = Parameter("p2", 2.0, limits=(1, 3), floating=False)
    return p1, p2

def test_parameter_initialization(parameters):
    p1, p2 = parameters
    assert p1.name == "p1"
    assert p1.value == 1.0
    assert p1.limits == (0, 5)
    assert p1.floating

    assert p2.name == "p2"
    assert p2.value == 2.0
    assert p2.limits == (1, 3)
    assert not p2.floating

def test_parameter_setters(parameters):
    p1, _ = parameters
    p1.value = 4.0
    assert p1.value == 4.0

    p1.limits = (2, 6)
    assert p1.limits == (2, 6)

    p1.floating = False
    assert not p1.floating

def test_parameter_operations(parameters):
    p1, p2 = parameters
    p3 = p1 + p2
    assert isinstance(p3, Parameter)
    assert p3.value == p1.value + p2.value

    p4 = p1 + 3.0
    assert isinstance(p4, Parameter)
    assert p4.value == p1.value + 3.0

    p4 = 3.0 + p1
    assert isinstance(p4, Parameter)
    assert p4.value ==  3.0 + p1.value

    p5 = p1 * p2
    assert isinstance(p5, Parameter)
    assert p5.value == p1.value * p2.value

    p6 = p1 * 3.0
    assert isinstance(p6, Parameter)
    assert p6.value == p1.value * 3.0

    p5 = 3.0 * p1
    assert isinstance(p5, Parameter)
    assert p5.value == 3.0 * p1.value

    p7 = p1 - p2
    assert isinstance(p7, Parameter)
    assert p7.value == p1.value - p2.value

    p8 = p1 - 3.0
    assert isinstance(p8, Parameter)
    assert p8.value == p1.value - 3.0

    p8 = 3.0 - p1
    assert isinstance(p8, Parameter)
    assert p8.value == 3.0 - p1.value

    p9 = p1 / p2
    assert isinstance(p9, Parameter)
    assert p9.value == p1.value / p2.value

    p10 = p1 / 3.0
    assert isinstance(p10, Parameter)
    assert p10.value == p1.value / 3.0

    p10 = 3.0 / p1
    assert isinstance(p10, Parameter)
    assert p10.value == 3.0 / p1.value

    p11 = -p1
    assert isinstance(p11, Parameter)
    assert p11.value == -p1.value

    assert +p1 is p1

    p12 = (p1 + p2) * p1 - 3.0 / p2
    assert isinstance(p12, Parameter)
    assert p12.value == (p1.value + p2.value) * p1.value - 3.0 / p2.value


def test_composed_parameter_structure(parameters):
    p1, p2 = parameters

    p3 = p1 * p2
    assert p3.kind is ParKind.PRODUCT
    assert p3.sources == (p1, p2)
    assert p3.is_composed()
    assert not p3.is_floating()

    p4 = 3.0 - p1
    assert p4.kind is ParKind.DIFFERENCE
    assert p4.sources[0].kind is ParKind.CONSTANT
    assert p4.sources[0].value == 3.0
    assert p4.sources[1] is p1

    p5 = -p1
    assert p5.kind is ParKind.NEGATION
    assert p5.sources == (p1,)


def test_composed_parameter_follows_sources(parameters):
    p1, p2 = parameters
    p3 = p1 + p2

    p1.value = 4.0
    assert p3.value == 4.0 + p2.value


def test_composed_parameter_is_read_only(parameters):
    p1, p2 = parameters
    p3 = p1 + p2

    with pytest.raises(ValueError):
        p3.value = 1.0
    with pytest.raises(ValueError):
        p3.limits = [0., 1.]
    with pytest.raises(ValueError):
        p3.floating = True


def test_parameter_invalid_operand(parameters):
    p1, _ = parameters

    with pytest.raises(TypeError):
        _ = p1 + "a"
    with pytest.raises(TypeError):
        _ = None * p1
