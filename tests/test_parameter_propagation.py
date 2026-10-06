"""Test suite for the evaluation and gradient of composed parameters."""
import pytest
from flarefly.components import Parameter

A_VAL = 2.
B_VAL = 0.5


def free_value(par):
    """Return the value of a free parameter"""
    return par.value


@pytest.fixture(name="pars")
def fixture_pars():
    """Create two free parameters."""
    return Parameter("a", value=A_VAL), Parameter("b", value=B_VAL)


@pytest.mark.parametrize("expression, expected_value, expected_der_a, expected_der_b", [
    (lambda a, b: a + b, A_VAL + B_VAL, 1., 1.),
    (lambda a, b: a - b, A_VAL - B_VAL, 1., -1.),
    (lambda a, b: a * b, A_VAL * B_VAL, B_VAL, A_VAL),
    (lambda a, b: a / b, A_VAL / B_VAL, 1. / B_VAL, -A_VAL / B_VAL**2),
])
def test_binary_kinds(pars, expression, expected_value, expected_der_a, expected_der_b):
    """Test value and derivatives of each binary composed kind."""
    a, b = pars
    par = expression(a, b)

    assert par.evaluate(free_value) == pytest.approx(expected_value)
    gradient = par.gradient(free_value)
    assert gradient[a] == pytest.approx(expected_der_a)
    assert gradient[b] == pytest.approx(expected_der_b)


def test_negation(pars):
    """Test value and derivative of a negated parameter."""
    a, _ = pars
    par = -a

    assert par.evaluate(free_value) == pytest.approx(-A_VAL)
    assert par.gradient(free_value) == {a: pytest.approx(-1.)}


def test_free_and_constant(pars):
    """Test that a free parameter has unit derivative and constants do not appear in the gradient."""
    a, _ = pars
    par = 3 * a

    assert a.gradient(free_value) == {a: 1.}
    assert par.gradient(free_value) == {a: pytest.approx(3.)}


def test_repeated_parameter(pars):
    """Test that the contributions of a parameter appearing more than once are summed."""
    a, b = pars
    par = (a + b) * a

    gradient = par.gradient(free_value)
    assert gradient[a] == pytest.approx(2 * A_VAL + B_VAL)
    assert gradient[b] == pytest.approx(A_VAL)


def test_evaluate_uses_free_value(pars):
    """Test that the values of the free parameters are taken from free_value."""
    a, b = pars
    par = a / b
    fitted = {a: 4., b: 2.}

    assert par.evaluate(fitted.get) == pytest.approx(2.)
    assert par.gradient(fitted.get)[b] == pytest.approx(-1.)


def test_nested_expression_finite_difference(pars):
    """Test the gradient of a nested expression against a numerical derivative."""
    a, b = pars
    par = (a * b - 1.5) / (a + 2 * b) - b / a
    step = 1.e-6

    gradient = par.gradient(free_value)
    for src in (a, b):
        def shifted(par_to_eval, shift, shifted_par=src):
            return par_to_eval.value + (shift if par_to_eval is shifted_par else 0.)
        numerical = (par.evaluate(lambda p: shifted(p, step)) - par.evaluate(lambda p: shifted(p, -step))) / (2 * step)
        assert gradient[src] == pytest.approx(numerical, rel=1.e-6)
