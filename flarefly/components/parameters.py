"""
Module defining the backend-agnostic fit parameter and its conversion to the fit backends.
"""
import operator
from enum import Enum
from numbers import Number

import zfit


class ParKind(Enum):
    """Enumeration of how a parameter is obtained"""
    FREE = "free"
    CONSTANT = "constant"
    SUM = "sum"
    DIFFERENCE = "difference"
    PRODUCT = "product"
    RATIO = "ratio"
    NEGATION = "negation"


# operation and symbol used to evaluate and print each composed kind
_OPERATIONS = {
    ParKind.SUM: (operator.add, "+"),
    ParKind.DIFFERENCE: (operator.sub, "-"),
    ParKind.PRODUCT: (operator.mul, "*"),
    ParKind.RATIO: (operator.truediv, "/"),
    ParKind.NEGATION: (operator.neg, "-"),
}

# partial derivatives of each composed kind with respect to its sources, given the source values
_DERIVATIVES = {
    ParKind.SUM: lambda a, b: (1., 1.),
    ParKind.DIFFERENCE: lambda a, b: (1., -1.),
    ParKind.PRODUCT: lambda a, b: (b, a),
    ParKind.RATIO: lambda a, b: (1. / b, -a / b**2),
    ParKind.NEGATION: lambda a: (-1.,),
}


class Parameter:  # pylint: disable=too-many-instance-attributes
    """
    Fit parameter.

    Parameters
    -------------------------------------------------
    name: str
        Name of the parameter
    value: float or None
        Initial value. None if not set yet. Ignored for composed parameters, whose value is derived
    sources: tuple
        Parameters this one is derived from. Empty for free and constant parameters
    kind: ParKind
        How the parameter is obtained
    floating: bool or None
        Whether the parameter is free to float in the fit. None if not set yet.
        Always False for composed and constant parameters
    limits: list or None
        Lower and upper limits of a free parameter, [lower, upper]. None if not set yet
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, name, value=None, sources=(), kind=ParKind.FREE, floating=True, limits=None):
        self.name = name
        self.kind = ParKind(kind)
        self.sources = tuple(sources)
        self._value = None if value is None else float(value)
        self._floating = floating if self.kind is ParKind.FREE else False
        self._limits = limits if self.kind is ParKind.FREE else None
        self._auto_name = False

    def __repr__(self):
        return (f"Parameter(name={self.name!r}, value={self.value}, kind={self.kind.name}, "
                f"floating={self.floating}, limits={self.limits}, "
                f"sources={[src.name for src in self.sources]})")

    def __str__(self):
        return self.name

    @property
    def value(self):
        """Get the current value, evaluating the sources for composed parameters"""
        if not self.is_composed():
            return self._value
        return _OPERATIONS[self.kind][0](*(src.value for src in self.sources))

    @value.setter
    def value(self, value):
        """Set the current value. Only allowed for free parameters"""
        self._check_free("value")
        self._value = None if value is None else float(value)

    @property
    def floating(self):
        """Get whether the parameter floats in the fit"""
        return self._floating

    @floating.setter
    def floating(self, value):
        """Set whether the parameter floats in the fit. Only allowed for free parameters"""
        self._check_free("floating flag")
        self._floating = value

    def is_floating(self):
        """Check whether the parameter floats in the fit"""
        return self.floating if self.kind is ParKind.FREE else False

    @property
    def limits(self):
        """Get the limits of the parameter"""
        return self._limits

    @limits.setter
    def limits(self, value):
        """Set the limits of the parameter. Only allowed for free parameters"""
        self._check_free("limits")
        self._limits = value

    @property
    def operation(self):
        """Get the callable combining the source values. None for free and constant parameters"""
        if not self.is_composed():
            return None
        return _OPERATIONS[self.kind][0]

    @property
    def auto_name(self):
        """Get whether the name was generated automatically when composing the parameter"""
        return self._auto_name

    def is_composed(self):
        """Check if the parameter is derived from other parameters"""
        return self.kind in _OPERATIONS

    def evaluate(self, free_value):
        """
        Evaluate the parameter, taking the value of each free parameter from free_value

        Parameters
        -------------------------------------------------
        free_value: callable
            Function returning the value of a free parameter

        Returns
        -------------------------------------------------
        value: float
            The value of the parameter
        """
        if self.kind is ParKind.FREE:
            return free_value(self)
        if self.kind is ParKind.CONSTANT:
            return self._value
        return self.operation(*(src.evaluate(free_value) for src in self.sources))  # pylint: disable=not-callable

    def gradient(self, free_value):
        """
        Get the derivatives with respect to the free parameters this parameter depends on

        Parameters
        -------------------------------------------------
        free_value: callable
            Function returning the value of a free parameter, at which the derivatives are evaluated

        Returns
        -------------------------------------------------
        gradient: dict
            Dictionary {free parameter: derivative}
        """
        if self.kind is ParKind.FREE:
            return {self: 1.}
        if self.kind is ParKind.CONSTANT:
            return {}
        values = [src.evaluate(free_value) for src in self.sources]
        gradient = {}
        for src, derivative in zip(self.sources, _DERIVATIVES[self.kind](*values)):
            for par, src_derivative in src.gradient(free_value).items():
                gradient[par] = gradient.get(par, 0.) + derivative * src_derivative
        return gradient

    def free_sources(self):
        """Get the free parameters this parameter ultimately depends on, without duplicates"""
        if self.kind is ParKind.FREE:
            return [self]
        free = []
        for src in self.sources:
            free.extend(par for par in src.free_sources() if par not in free)
        return free

    def _check_free(self, what):
        """Refuse to modify a field that is derived for composed and constant parameters"""
        if self.kind is not ParKind.FREE:
            raise ValueError(f"Cannot set the {what} of {self.kind.value} parameter '{self.name}'")

    @staticmethod
    def _as_parameter(other):
        """Wrap a number into a constant parameter"""
        if isinstance(other, Parameter):
            return other
        if isinstance(other, Number):
            return Parameter(repr(other), value=other, kind=ParKind.CONSTANT, floating=False)
        return NotImplemented

    @staticmethod
    def _compose(kind, *sources):
        """Build a composed parameter from its sources"""
        symbol = _OPERATIONS[kind][1]
        if len(sources) == 1:
            name = f"{symbol}{sources[0].name}"
        else:
            name = f"({sources[0].name}{symbol}{sources[1].name})"
        par = Parameter(name, sources=sources, kind=kind, floating=False)
        par._auto_name = True  # pylint: disable=protected-access
        return par

    def _binary(self, other, kind, reflected=False):
        """Compose self with other, in reversed order for reflected operators"""
        other = self._as_parameter(other)
        if other is NotImplemented:
            return NotImplemented
        return self._compose(kind, other, self) if reflected else self._compose(kind, self, other)

    def __add__(self, other):
        return self._binary(other, ParKind.SUM)

    def __radd__(self, other):
        return self._binary(other, ParKind.SUM, reflected=True)

    def __sub__(self, other):
        return self._binary(other, ParKind.DIFFERENCE)

    def __rsub__(self, other):
        return self._binary(other, ParKind.DIFFERENCE, reflected=True)

    def __mul__(self, other):
        return self._binary(other, ParKind.PRODUCT)

    def __rmul__(self, other):
        return self._binary(other, ParKind.PRODUCT, reflected=True)

    def __truediv__(self, other):
        return self._binary(other, ParKind.RATIO)

    def __rtruediv__(self, other):
        return self._binary(other, ParKind.RATIO, reflected=True)

    def __neg__(self):
        return self._compose(ParKind.NEGATION, self)

    def __pos__(self):
        return self

    def __float__(self):
        return float(self.value)


class ZfitParameterConverter:  # pylint: disable=too-few-public-methods
    """
    Class converting flarefly parameters into zfit parameters.

    Conversions are cached per Parameter object, so a parameter shared between PDFs
    becomes a single zfit parameter. Use one converter per model build.
    """

    def __init__(self):
        self._cache = {}

    def convert(self, par):
        """
        Convert a flarefly parameter, and recursively its sources, into a zfit parameter.

        Parameters
        -------------------------------------------------
        par: Parameter
            The parameter to convert

        Returns
        -------------------------------------------------
        zfit_par: zfit.Parameter, zfit.param.ConstantParameter or zfit.ComposedParameter
            The corresponding zfit parameter
        """
        if id(par) in self._cache:
            return self._cache[id(par)][1]

        if par.kind is ParKind.FREE:
            lower, upper = par.limits if par.limits is not None else (None, None)
            zfit_par = zfit.Parameter(
                par.name, par.value, lower, upper,
                floating=par.floating if par.floating is not None else True
            )
        elif par.kind is ParKind.CONSTANT:
            zfit_par = zfit.param.ConstantParameter(par.name, par.value)
        else:
            zfit_par = zfit.ComposedParameter(
                par.name, par.operation,
                params=[self.convert(src) for src in par.sources],
                unpack_params=True
            )

        # keep a reference to par, so that its id is not reused while cached
        self._cache[id(par)] = (par, zfit_par)
        return zfit_par
