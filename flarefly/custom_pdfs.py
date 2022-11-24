"""
Module containing custom pdfs
"""

import zfit
import tensorflow as tf
from zfit import z
from scipy.special import voigt_profile

# pylint: disable=too-many-ancestors


class DoubleGauss(zfit.pdf.ZPDF):
    """
    PDF composed by the sum of two gaussians sharing the same mean parameter
    Parameters:

        - mu: shared mean parameter

        - sigma1: sigma of first Gaussian function

        - sigma2: sigma of second Gaussian function

        - frac1: fraction of integral associated to first Gaussian function
    """

    # override the name of the parameters
    _PARAMS = ['mu', 'sigma1', 'sigma2', 'frac1']

    def _unnormalized_pdf(self, x):
        """
        PDF 'unnormalized'.
        See https://zfit.github.io/zfit/_modules/zfit/core/basepdf.html#BasePDF.unnormalized_pdf
        for more details
        """
        x = zfit.z.unstack_x(x)
        mean = self.params['mu']
        sigma1 = self.params['sigma1']
        sigma2 = self.params['sigma2']
        frac1 = self.params['frac1']
        gauss1 = tf.exp(- ((x - mean)/sigma1)**2)
        gauss2 = tf.exp(- ((x - mean)/sigma2)**2)
        return frac1 * gauss1 + (1-frac1) * gauss2


class Pow(zfit.pdf.ZPDF):
    """
    PDF composed by a power-law function
    f(x) = (x - m)^power

    Parameters:

        - mass: mass parameter

        - power: exponential power
    """

    # override the name of the parameters
    _PARAMS = ['mass', 'power']

    def _unnormalized_pdf(self, x):
        """
        PDF 'unnormalized'.
        See https://zfit.github.io/zfit/_modules/zfit/core/basepdf.html#BasePDF.unnormalized_pdf
        for more details
        """
        x = zfit.z.unstack_x(x)
        mass = self.params['mass']
        power = self.params['power']
        return tf.pow(x - mass, power)


class ExpoPow(zfit.pdf.ZPDF):
    """
    PDF composed by an exponential power-law function
    f(x) = sqrt(x - m) * exp( -lam * (x - m) )

    Parameters:

        - mass: mass parameter

        - lam: slope of exponential
    """

    # override the name of the parameters
    _PARAMS = ['mass', 'lam']

    @tf.function
    def _unnormalized_pdf(self, x):
        """
        PDF 'unnormalized'.
        See https://zfit.github.io/zfit/_modules/zfit/core/basepdf.html#BasePDF.unnormalized_pdf
        for more details
        """
        x = zfit.z.unstack_x(x)
        mass = self.params['mass']
        lam = self.params['lam']
        return tf.sqrt(x - mass) * tf.exp(-lam * (x - mass))

class Voigtian(zfit.pdf.ZPDF):
    """
    Voigtian PDF defined starting from the scipy.special definition
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.voigt_profile.html
    for more details

    Parameters:

        - mu: mass parameter

        - sigma: sigma of the gaussian

        - gamma: gamma of the cauchy
    """

    # override the name of the parameters
    _PARAMS = ['mu', 'sigma', 'gamma']

    def _unnormalized_pdf(self, x):
        """
        PDF 'unnormalized'.
        See https://zfit.github.io/zfit/_modules/zfit/core/basepdf.html#BasePDF.unnormalized_pdf
        for more details
        """
        zfit.run.assert_executing_eagerly()  # make sure we're eager
        x = zfit.z.unstack_x(x)
        mass = self.params['mu']
        gamma = self.params['gamma']
        sigma = self.params['sigma']

        return z.convert_to_tensor(voigt_profile(x - mass, sigma, gamma))
