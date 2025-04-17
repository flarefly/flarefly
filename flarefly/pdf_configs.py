"""Helper functions to get the PDF configurations for signal and background PDFs."""
import zfit
import pdg
import flarefly.custom_pdfs as cpdf
from flarefly.utils import Logger

pdg_api = pdg.connect()

SIGNAL_PDF_CONFIGS = {
    'gaussian': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigma': {
                'init': 0.010,
                'limits': [0., None],
            }
        },
        'pdf_class': zfit.pdf.Gauss,
        'pdf_args': ['mu', 'sigma']
    },
    'doublegaus': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigma1': {
                'init': 0.010,
                'limits': [0., None],
            },
            'sigma2': {
                'init': 0.100,
                'limits': [0., None],
            },
            'frac1': {
                'init': 0.9,
                'limits': [0., 1.],
            }
        },
        'pdf_class': cpdf.DoubleGauss,  # Assuming cpdf is imported
        'pdf_args': ['mu', 'sigma1', 'sigma2', 'frac1']
    },
    'crystalball': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigma': {
                'init': 0.010,
                'limits': [0., None],
            },
            'alpha': {
                'init': 0.5,
                'limits': [0, None],
            },
            'n': {
                'init': 1.,
                'limits': [0., None],
            }
        },
        'pdf_class': zfit.pdf.CrystalBall,
        'pdf_args': ['mu', 'sigma', 'alpha', 'n']
    },
    'doublecb': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigma': {
                'init': 0.010,
                'limits': [0., None],
            },
            'alphal': {
                'init': 0.5,
                'limits': [0, None],
            },
            'nl': {
                'init': 1.,
                'limits': [0., None],
            },
            'alphar': {
                'init': 0.5,
                'limits': [0, None],
            },
            'nr': {
                'init': 1.,
                'limits': [0., None],
            }
        },
        'pdf_class': zfit.pdf.DoubleCB,
        'pdf_args': ['mu', 'sigma', 'alphal', 'nl', 'alphar', 'nr']
    },
    'doublecbsymm': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigma': {
                'init': 0.010,
                'limits': [0., None],
            },
            'alpha': {
                'init': 0.5,
                'limits': [0, None],
            },
            'n': {
                'init': 1.,
                'limits': [0., None],
            }
        },
        'args_mapping': {
            'mu': 'mu',
            'sigma': 'sigma',
            'nl': 'n',
            'nr': 'n',
            'alphar': 'alpha',
            'alphal': 'alpha'
        },
        'pdf_class': zfit.pdf.DoubleCB,
        'pdf_args': ['mu', 'sigma', 'alphal', 'nl', 'alphar', 'nr']
    },
    'cauchy': {
        'parameters': {
            'm': {
                'init': 1.865,
                'limits': [0., None],
            },
            'gamma': {
                'init': 0.010,
                'limits': [0., None],
            }
        },
        'pdf_class': zfit.pdf.Cauchy,
        'pdf_args': ['m', 'gamma']
    },
    'voigtian': {
        'parameters': {
            'm': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigma': {
                'init': 0.010,
                'limits': [0., None],
            },
            'gamma': {
                'init': 0.010,
                'limits': [0., None],
            }
        },
        'pdf_class': zfit.pdf.Voigt,
        'pdf_args': ['m', 'sigma', 'gamma']
    },
    'gausexptail': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigma': {
                'init': 0.010,
                'limits': [0., None],
            },
            'alpha': {
                'init': 1.e6,
                'limits': [None, None],
            }
        },
        'pdf_class': zfit.pdf.GaussExpTail,
        'pdf_args': ['mu', 'sigma', 'alpha']
    },
    'genergausexptail': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigmal': {
                'init': 0.010,
                'limits': [0., None],
            },
            'sigmar': {
                'init': 0.010,
                'limits': [0., None],
            },
            'alphal': {
                'init': 1.e6,
                'limits': [None, None],
            },
            'alphar': {
                'init': 1.e6,
                'limits': [None, None],
            }
        },
        'pdf_class': zfit.pdf.GeneralizedGaussExpTail,
        'pdf_args': ['mu', 'sigmar', 'sigmal', 'alphar', 'alphal']
    },
    'genergausexptailsymm': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigma': {
                'init': 0.010,
                'limits': [0., None],
            },
            'alpha': {
                'init': 1.e6,
                'limits': [None, None],
            }
        },
        'args_mapping': {
            'mu': 'mu',
            'sigmar': 'sigma',
            'sigmal': 'sigma',
            'alphar': 'alpha',
            'alphal': 'alpha'
        },
        'pdf_class': zfit.pdf.GeneralizedGaussExpTail,
        'pdf_args': ['mu', 'sigmar', 'sigmal', 'alphar', 'alphal']
    },
    'bifurgaus': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigmal': {
                'init': 0.010,
                'limits': [0., None],
            },
            'sigmar': {
                'init': 0.010,
                'limits': [0., None],
            }
        },
        'pdf_class': zfit.pdf.BifurGauss,
        'pdf_args': ['mu', 'sigmar', 'sigmal']
    },
    'genercrystalball': {
        'parameters': {
            'mu': {
                'init': 1.865,
                'limits': [0., None],
            },
            'sigmal': {
                'init': 0.010,
                'limits': [0., None],
            },
            'sigmar': {
                'init': 0.010,
                'limits': [0., None],
            },
            'alphal': {
                'init': 0.5,
                'limits': [0, None],
            },
            'nl': {
                'init': 1.,
                'limits': [0., None],
            },
            'alphar': {
                'init': 0.5,
                'limits': [0, None],
            },
            'nr': {
                'init': 1.,
                'limits': [0., None],
            }
        },
        'pdf_class': zfit.pdf.GeneralizedCB,
        'pdf_args': ['mu', 'sigmar', 'sigmal', 'alphal', 'nl', 'alphar', 'nr']
    }
}

BACKGROUND_PDF_CONFIGS = {
    'chebpol': {
        'parameters': {
            'c': {
                'init': 0.1,
                'limits': [None, None]
            }
        },
        'pdf_class': zfit.pdf.Exponential,
        'pdf_args': ['coeff0', 'coeffs']
    },
    'expo': {
        'parameters': {
            'lam': {
                'init': 0.1,
                'limits': [None, None]
            }
        },
        'pdf_class': zfit.pdf.Exponential,
        'pdf_args': ['lam']
    },
    'powlaw': {
        'parameters': {
            'mass': {
                'init': pdg_api.get_particle_by_mcid(211).mass,  # pion mass
                'limits': [0., None]
            },
            'power': {
                'init': 1.,
                'limits': [None, None]
            }
        },
        'pdf_class': cpdf.Pow,
        'pdf_args': ['mass', 'power']
    },
    'expopow': {
        'parameters': {
            'mass': {
                'init': pdg_api.get_particle_by_mcid(211).mass,
                'limits': [0., None]
            },
            'lam': {
                'init': 0.1,
                'limits': [None, None]
            }
        },
        'pdf_class': cpdf.ExpoPow,
        'pdf_args': ['mass', 'lam']
    },
    'expopowext': {
        'parameters': {
            'mass': {
                'init': pdg_api.get_particle_by_mcid(211).mass,
                'limits': [0., None]
            },
            'power': {
                'init': 0.5,
                'limits': [0., None]
            },
            'c1': {
                'init': -0.1,
                'limits': [None, None]
            },
            'c2': {
                'init': 0.,
                'limits': [None, None]
            },
            'c3': {
                'init': 0.,
                'limits': [None, None]
            }
        },
        'pdf_class': cpdf.ExpoPowExt,
        'pdf_args': ['mass', 'power', 'c1', 'c2', 'c3']
    }
}

KDE_MAP = {
    'kde_exact': zfit.pdf.KDE1DimExact,
    'kde_grid': zfit.pdf.KDE1DimGrid,
    'kde_fft': zfit.pdf.KDE1DimFFT,
    'kde_isj': zfit.pdf.KDE1DimISJ
}


def get_signal_pdf_config(pdf_name: str):
    """Get the configuration for a specific PDF."""
    if pdf_name not in SIGNAL_PDF_CONFIGS:
        Logger(f"Unknown signal PDF name: {pdf_name}.", "FATAL")

    # Get the specific config
    config = SIGNAL_PDF_CONFIGS[pdf_name].copy()
    config.update({'floating': False})  # Default to not floating

    return config


def get_bkg_pdf_config(pdf_name: str):
    """Get the configuration for a specific PDF."""
    if pdf_name not in BACKGROUND_PDF_CONFIGS:
        Logger(f"Unknown background PDF name: {pdf_name}.", "FATAL")

    # Get the specific config
    config = BACKGROUND_PDF_CONFIGS[pdf_name].copy()
    config.update({'floating': False})  # Default to not floating

    return config


def get_kde_pdf(kde_name: str):
    """Get the KDE PDF class for a specific KDE type."""
    if kde_name not in KDE_MAP:
        Logger(
            f"Unknown KDE type: {kde_name}. "
            f"Available types: {list(KDE_MAP.keys())}",
            "FATAL"
        )

    return KDE_MAP[kde_name]
