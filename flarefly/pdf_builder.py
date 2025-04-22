"""PDFBuilder class to create signal and background PDFs using zfit."""
from typing import Dict, Any, List, Optional, Tuple
import zfit
from flarefly.pdf_configs import get_signal_pdf_config, get_bkg_pdf_config, get_kde_pdf
from flarefly.utils import Logger


class PDFBuilder:
    """Class to build signal and background PDFs using zfit."""
    @staticmethod
    def build_signal_pdf(
        pdf_name: str,
        obs: zfit.Space,
        name: str,
        ipdf: int,
        init_pars: Dict[str, float],
        limits_pars: Dict[str, List[float]],
        fix_pars: Dict[str, bool]
    ) -> zfit.pdf.BasePDF:
        """Build a signal PDF with configurable parameters.

        Args:
            pdf_name: Name of the PDF to build
            obs: The observable space for the PDF
            name: Base name for parameters
            ipdf: Index of the PDF
            init_pars: Dictionary that will be updated with default initial values
            limits_pars: Dictionary that will be updated with default limits
            fix_pars: Dictionary that will be updated with default fix flags

        Returns:
            - The constructed zfit PDF object
            - Dictionary of zfit.Parameter objects for the PDF parameters
        """
        config = get_signal_pdf_config(pdf_name)

        # Update the input dictionaries with default values
        PDFBuilder._update_with_defaults(config, init_pars, limits_pars, fix_pars)

        parameters = {}
        for par_name in config['parameters'].keys():
            param_name = f'{name}_{par_name}_signal{ipdf}'
            parameters[param_name] = zfit.Parameter(
                name=param_name,
                value=init_pars[par_name],
                lower=limits_pars[par_name][0],
                upper=limits_pars[par_name][1],
                floating=not fix_pars[par_name]
            )

        pdf_args = {'obs': obs}
        for arg_name in config['pdf_args']:
            if 'args_mapping' in config:
                param_key = f"{name}_{config['args_mapping'][arg_name]}_signal{ipdf}"
            else:
                param_key = f'{name}_{arg_name}_signal{ipdf}'
            pdf_args[arg_name] = parameters[param_key]

        return config['pdf_class'](**pdf_args), parameters

    @staticmethod
    def build_signal_kde(
        pdf_name: str,
        kde_sample: Any,
        name: str,
        ipdf: int,
        kde_options: Optional[Dict[str, Any]] = None
    ) -> zfit.pdf.BasePDF:
        """Build a signal KDE PDF.

        Args:
            pdf_name: Type of KDE ('kde_exact', 'kde_grid', 'kde_fft', 'kde_isj')
            kde_sample: The sample data for KDE estimation
            name: Base name for the PDF
            ipdf: Index of the PDF
            kde_options: Additional options for the KDE

        Returns:
            The constructed KDE PDF
        """
        if not kde_sample:
            Logger(f'Missing datasample for Kernel Density Estimation of signal {ipdf}!', 'FATAL')

        kde_options = kde_options or {}

        return get_kde_pdf(pdf_name)(
            data=kde_sample.get_data(),
            obs=kde_sample.get_obs(),
            name=f'{name}_kde_signal{ipdf}',
            **kde_options
        )

    @staticmethod
    def build_signal_hist(
        obs: zfit.Space,
        hist_signal_sample: Any,
        name: str,
        ipdf: int,
    ) -> zfit.pdf.BasePDF:
        """Build a signal PDF from a histogram template.

        Args:
            hist_signal_sample: The histogram
            name: Base name for the PDF
            ipdf: Index of the PDF

        Returns:
            The constructed PDF
        """
        if not hist_signal_sample:
            Logger(f'Missing datasample for histogram template of signal {ipdf}!', 'FATAL')

        return zfit.pdf.SplinePDF(
            zfit.pdf.HistogramPDF(
                hist_signal_sample.get_binned_data(),
                name=f'{name}_hist_signal{ipdf}'
            ),
            order=3,
            obs=obs
        )

    @staticmethod
    def build_bkg_pdf(
        pdf_name: str,
        obs: zfit.Space,
        name: str,
        ipdf: int,
        init_pars: Dict[str, float],
        limits_pars: Dict[str, List[float]],
        fix_pars: Dict[str, bool]
    ) -> zfit.pdf.BasePDF:
        """Build a signal PDF with configurable parameters.

        Args:
            pdf_name: Name of the PDF to build
            obs: The observable space for the PDF
            name: Base name for parameters
            ipdf: Index of the PDF
            init_pars: Dictionary that will be updated with default initial values
            limits_pars: Dictionary that will be updated with default limits
            fix_pars: Dictionary that will be updated with default fix flags

        Returns:
            - The constructed zfit PDF object
            - Dictionary of zfit.Parameter objects for the PDF parameters
        """
        # Handle Chebyshev polynomials specially
        if 'chebpol' in pdf_name:
            return PDFBuilder._build_chebyshev_pdf(
                pdf_name, obs, name, ipdf, init_pars, limits_pars, fix_pars
            )

        config = get_bkg_pdf_config(pdf_name)

        # Update the input dictionaries with default values
        PDFBuilder._update_with_defaults(config, init_pars, limits_pars, fix_pars)

        parameters = {}
        for par_name in config['parameters'].keys():
            param_name = f'{name}_{par_name}_bkg{ipdf}'
            parameters[param_name] = zfit.Parameter(
                name=param_name,
                value=init_pars[par_name],
                lower=limits_pars[par_name][0],
                upper=limits_pars[par_name][1],
                floating=not fix_pars[par_name]
            )

        pdf_args = {'obs': obs}
        for arg_name in config['pdf_args']:
            param_key = f'{name}_{arg_name}_bkg{ipdf}'
            pdf_args[arg_name] = parameters[param_key]

        return config['pdf_class'](**pdf_args), parameters

    @staticmethod
    def _build_chebyshev_pdf(
        pdf_name: str,
        obs: zfit.Space,
        name: str,
        ipdf: int,
        init_pars: Dict[str, float],
        limits_pars: Dict[str, List[float]],
        fix_pars: Dict[str, bool]
    ) -> Tuple[zfit.pdf.BasePDF, Dict[str, zfit.Parameter]]:
        """Build a Chebyshev polynomial background PDF."""
        pol_degree = int(pdf_name.split('chebpol')[1])

        # Create parameters for each coefficient
        parameters = {}
        for deg in range(pol_degree + 1):
            par_name = f'c{deg}'
            init_pars.setdefault(par_name, 0.1)
            limits_pars.setdefault(par_name, [None, None])
            fix_pars.setdefault(par_name, False)

            param_name = f'{name}_{par_name}_bkg{ipdf}'
            parameters[param_name] = zfit.Parameter(
                name=param_name,
                value=init_pars[par_name],
                lower=limits_pars[par_name][0],
                upper=limits_pars[par_name][1],
                floating=not fix_pars[par_name]
            )

        # Prepare Chebyshev arguments
        coeff0 = parameters[f'{name}_c0_bkg{ipdf}']
        bkg_coeffs = [parameters[f'{name}_c{deg}_bkg{ipdf}'] for deg in range(1, pol_degree + 1)]

        return zfit.pdf.Chebyshev(obs=obs, coeff0=coeff0, coeffs=bkg_coeffs), parameters

    @staticmethod
    def build_bkg_kde(
        pdf_name: str,
        kde_sample: Any,
        name: str,
        ipdf: int,
        kde_options: Optional[Dict[str, Any]] = None
    ) -> zfit.pdf.BasePDF:
        """Build a background KDE PDF.

        Args:
            pdf_name: Type of KDE ('kde_exact', 'kde_grid', 'kde_fft', 'kde_isj')
            kde_sample: The sample data for KDE estimation
            name: Base name for the PDF
            ipdf: Index of the PDF
            kde_options: Additional options for the KDE

        Returns:
            The constructed KDE PDF
        """
        if not kde_sample:
            Logger(f'Missing datasample for Kernel Density Estimation of background {ipdf}!', 'FATAL')

        kde_options = kde_options or {}

        return get_kde_pdf(pdf_name)(
            data=kde_sample.get_data(),
            obs=kde_sample.get_obs(),
            name=f'{name}_kde_bkg{ipdf}',
            **kde_options
        )

    @staticmethod
    def build_bkg_hist(
        obs: zfit.Space,
        hist_bkg_sample: Any,
        name: str,
        ipdf: int,
    ) -> zfit.pdf.BasePDF:
        """Build a bkg PDF from a histogram template.

        Args:
            hist_bkg_sample: The histogram
            name: Base name for the PDF
            ipdf: Index of the PDF

        Returns:
            The constructed PDF
        """
        if not hist_bkg_sample:
            Logger(f'Missing datasample for histogram template of background {ipdf}!', 'FATAL')

        return zfit.pdf.SplinePDF(
            zfit.pdf.HistogramPDF(
                hist_bkg_sample.get_binned_data(),
                name=f'{name}_hist_bkg{ipdf}'
            ),
            order=3,
            obs=obs
        )

    @staticmethod
    def _update_with_defaults(
        config: Dict[str, Any],
        init_pars: Dict[str, float],
        limits_pars: Dict[str, List[float]],
        fix_pars: Dict[str, bool]
    ) -> None:
        """Update the parameter dictionaries with default values from config."""
        for par_name, par_config in config['parameters'].items():
            init_pars.setdefault(par_name, par_config['init'])
            limits_pars.setdefault(par_name, par_config['limits'])
            fix_pars.setdefault(par_name, par_config['fix'])
