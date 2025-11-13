"""PDFBuilder class to create signal and background PDFs using zfit."""
from typing import Dict, Any, List, Optional, Tuple
import zfit
import pdg
from flarefly.pdf_configs import get_signal_pdf_config, get_bkg_pdf_config, get_kde_pdf
from flarefly.utils import Logger
import flarefly.custom_pdfs as cpdf
from flarefly.components.pdf_kind import PDFKind, PDFType
from flarefly.components.pdf_base import F2PDFBase

class PDFBuilder:
    """Class to build signal and background PDFs using zfit."""

    def __init__(self):
        self.pdg_api = pdg.connect()

    @staticmethod
    def build_signal_pdf(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
    ) -> zfit.pdf.BasePDF:
        """Build a signal PDF with configurable parameters.

        Args:
            pdf: The PDF to build
            obs: The observable space for the PDF
            name: Base name for parameters
            ipdf: Index of the PDF
        """
        pdf.print()
        if pdf.is_kde():
            pdf = PDFBuilder.build_signal_kde(
                pdf,
                name,
                ipdf
            )
            return
        if pdf.is_hist():
            pdf = PDFBuilder.build_signal_hist(
                pdf,
                obs,
                name,
                ipdf
            )
            return

        config = get_signal_pdf_config(pdf.kind)

        # Update the input dictionaries with default values
        PDFBuilder._update_with_defaults(config, pdf)

        parameters = {}
        for par_name in config['parameters'].keys():
            param_name = f'{name}_{par_name}_signal{ipdf}'
            parameters[param_name] = zfit.Parameter(
                name=param_name,
                value=pdf.get_init_par(par_name),
                lower=pdf.get_limits_par(par_name)[0],
                upper=pdf.get_limits_par(par_name)[1],
                floating=not pdf.get_fix_par(par_name)
            )

        pdf_args = {'obs': obs}
        for arg_name in config['pdf_args']:
            if 'args_mapping' in config: # flarefly -> zfit argument name mapping
                param_key = f"{name}_{config['args_mapping'][arg_name]}_signal{ipdf}"
            else:
                param_key = f'{name}_{arg_name}_signal{ipdf}'
            pdf_args[arg_name] = parameters[param_key]

        pdf.set_pdf(config['pdf_class'](**pdf_args))
        pdf.parameters = parameters

        if pdf.at_threshold:
            # pion mass as default
            pdf.set_default_init_par('powerthr', 1.)
            pdf.set_default_init_par('massthr', pdf.pdg_api.get_particle_by_mcid(211).mass)
            pdf.set_default_limits_par('massthr', [None, None])
            pdf.set_default_limits_par('powerthr', [None, None])
            pdf.set_default_fix_par('powerthr', False)
            pdf.set_default_fix_par('massthr', True)
            pdf.parameters[f'{name}_massthr_signal{ipdf}'] = zfit.Parameter(
                f'{name}_massthr_signal{ipdf}', pdf.get_init_par('massthr'),
                pdf.get_limits_par('massthr')[0], pdf.get_limits_par('massthr')[1],
                floating=not pdf.get_fix_par('massthr'))
            pdf.parameters[f'{name}_powerthr_signal{ipdf}'] = zfit.Parameter(
                f'{name}_powerthr_signal{ipdf}', pdf.get_init_par('powerthr'),
                pdf.get_limits_par('powerthr')[0], pdf.get_limits_par('powerthr')[1],
                floating=not pdf.get_fix_par('powerthr'))
            signalthr_pdf = cpdf.Pow(
                obs=obs,
                mass=pdf.parameters[f'{name}_massthr_signal{ipdf}'],
                power=pdf.parameters[f'{name}_powerthr_signal{ipdf}']
            )
            pdf.set_pdf(zfit.pdf.ProductPDF(
                [pdf.get_pdf(), signalthr_pdf], obs=obs))

    @staticmethod
    def build_signal_kde(
        pdf: F2PDFBase,
        name: str,
        ipdf: int,
    ) -> zfit.pdf.BasePDF:
        """Build a signal KDE PDF.

        Args:
            pdf: The PDF to build
            name: Base name for the PDF
            ipdf: Index of the PDF
        """
        if not pdf.kde_sample:
            Logger(f'Missing datasample for Kernel Density Estimation of signal {ipdf}!', 'FATAL')

        kde_options = pdf.kde_option or {}

        pdf.set_pdf(get_kde_pdf(pdf.kind)(
            data=pdf.kde_sample,
            obs=pdf.kde_sample.get_obs(),
            name=f'{name}_kde_signal{ipdf}',
            **kde_options
        ))

    @staticmethod
    def build_signal_hist(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
    ) -> zfit.pdf.BasePDF:
        """Build a signal PDF from a histogram template.

        Args:
            pdf: The PDF to build
            obs: The observable space for the PDF
            name: Base name for the PDF
            ipdf: Index of the PDF

        Returns:
            The constructed PDF
        """
        if not pdf.hist_sample:
            Logger(f'Missing datasample for histogram template of signal {ipdf}!', 'FATAL')

        pdf.set_pdf(zfit.pdf.SplinePDF(
            zfit.pdf.HistogramPDF(
                pdf.hist_sample.get_binned_data(),
                name=f'{name}_hist_signal{ipdf}'
            ),
            order=3,
            obs=obs
        ))

    @staticmethod
    def build_bkg_pdf(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
    ) -> zfit.pdf.BasePDF:
        """Build a background PDF with configurable parameters.

        Args:
            pdf: The PDF to build
            obs: The observable space for the PDF
            name: Base name for parameters
            ipdf: Index of the PDF
        """
        if pdf.is_kde():
            pdf = PDFBuilder.build_bkg_kde(
                pdf,
                name,
                ipdf
            )
            return
        if pdf.is_hist():
            pdf = PDFBuilder.build_bkg_hist(
                pdf,
                obs,
                name,
                ipdf
            )
            return
        if pdf.kind == PDFType.CHEBPOL:
            # Handle Chebyshev polynomials specially
            print("is reaally chebpol")
            PDFBuilder._build_chebyshev_pdf(
                pdf, obs, name, ipdf
            )
            return

        config = get_bkg_pdf_config(pdf.kind)

        # Update the input dictionaries with default values
        PDFBuilder._update_with_defaults(config, pdf)


        parameters = {}
        for par_name in config['parameters'].keys():
            param_name = f'{name}_{par_name}_signal{ipdf}'
            parameters[param_name] = zfit.Parameter(
                name=param_name,
                value=pdf.get_init_par(par_name),
                lower=pdf.get_limits_par(par_name)[0],
                upper=pdf.get_limits_par(par_name)[1],
                floating=not pdf.get_fix_par(par_name)
            )

        pdf_args = {'obs': obs}
        for arg_name in config['pdf_args']:
            if 'args_mapping' in config: # flarefly -> zfit argument name mapping
                param_key = f"{name}_{config['args_mapping'][arg_name]}_signal{ipdf}"
            else:
                param_key = f'{name}_{arg_name}_signal{ipdf}'
            pdf_args[arg_name] = parameters[param_key]

        pdf.set_pdf(config['pdf_class'](**pdf_args))
        pdf.parameters = parameters

    @staticmethod
    def _build_chebyshev_pdf(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
    ) -> Tuple[zfit.pdf.BasePDF, Dict[str, zfit.Parameter]]:
        """Build a Chebyshev polynomial background PDF."""

        # Create parameters for each coefficient
        parameters = {}
        for deg in range(pdf.kind.order + 1):
            par_name = f'c{deg}'
            pdf.set_default_init_par(par_name, 0.1)
            pdf.set_default_limits_par(par_name, [None, None])
            pdf.set_default_fix_par(par_name, False)

            param_name = f'{name}_{par_name}_bkg{ipdf}'
            parameters[param_name] = zfit.Parameter(
                name=param_name,
                value=pdf.get_init_par(par_name),
                lower=pdf.get_limits_par(par_name)[0],
                upper=pdf.get_limits_par(par_name)[1],
                floating=not pdf.get_fix_par(par_name)
            )

        # Prepare Chebyshev arguments
        coeff0 = parameters[f'{name}_c0_bkg{ipdf}']
        bkg_coeffs = [parameters[f'{name}_c{deg}_bkg{ipdf}'] for deg in range(1, pdf.kind.order + 1)]


        print("Parameters: chebyyy ", parameters)
        pdf.set_pdf(zfit.pdf.Chebyshev(obs=obs, coeff0=coeff0, coeffs=bkg_coeffs))
        pdf.parameters = parameters

        print(pdf.parameters)

    @staticmethod
    def build_bkg_kde(
        pdf: F2PDFBase,
        name: str,
        ipdf: int,
    ) -> zfit.pdf.BasePDF:
        """Build a background KDE PDF.

        Args:
            pdf_kind: Kind of KDE ('kde_exact', 'kde_grid', 'kde_fft', 'kde_isj')
            kde_sample: The sample data for KDE estimation
            name: Base name for the PDF
            ipdf: Index of the PDF
            kde_options: Additional options for the KDE

        Returns:
            The constructed KDE PDF
        """
        if not pdf.kde_sample:
            Logger(f'Missing datasample for Kernel Density Estimation of background {ipdf}!', 'FATAL')

        kde_options = kde_options or {}


        pdf.set_pdf(get_kde_pdf(pdf.kind)(
            data=pdf.kde_sample,
            obs=pdf.kde_sample.get_obs(),
            name=f'{name}_kde_signal{ipdf}',
            **kde_options
        ))

    @staticmethod
    def build_bkg_hist(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
    ) -> zfit.pdf.BasePDF:
        """Build a bkg PDF from a histogram template.

        Args:
            pdf: The PDF to build
            obs: The observable space for the PDF
            name: Base name for the PDF
            ipdf: Index of the PDF
        """
        if not pdf.hist_bkg_sample:
            Logger(f'Missing datasample for histogram template of background {ipdf}!', 'FATAL')

        pdf.set_pdf(zfit.pdf.SplinePDF(
            zfit.pdf.HistogramPDF(
                pdf.hist_bkg_sample.get_binned_data(),
                name=f'{name}_hist_bkg{ipdf}'
            ),
            order=3,
            obs=obs
        ))

    @staticmethod
    def _update_with_defaults(
        config: Dict[str, Any],
        pdf: F2PDFBase
    ) -> None:
        """Update the parameter dictionaries with default values from config."""
        init_pars = pdf.get_init_pars()
        limits_pars = pdf.get_limits_pars()
        fix_pars = pdf.get_fix_pars()
        for par_name, par_config in config['parameters'].items():
            init_pars.setdefault(par_name, par_config['init'])
            limits_pars.setdefault(par_name, par_config['limits'])
            fix_pars.setdefault(par_name, par_config['fix'])
