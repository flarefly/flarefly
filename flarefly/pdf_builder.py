"""PDFBuilder class to create signal and background PDFs using zfit."""
import zfit
from flarefly.pdf_configs import get_signal_pdf_config, get_bkg_pdf_config, get_kde_pdf
from flarefly.utils import Logger
import flarefly.custom_pdfs as cpdf
from flarefly.components.pdf_kind import PDFType
from flarefly.components.pdf_base import F2PDFBase
from flarefly.components.parameters import ZfitParameterConverter


class PDFBuilder:
    """Class to build signal and background PDFs using zfit."""

    @staticmethod
    def build_signal_pdf(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
        converter: ZfitParameterConverter,
    ):
        """Build a signal PDF with configurable parameters.

        Args:
            pdf: The PDF to build
            obs: The observable space for the PDF
            name: Base name for parameters
            ipdf: Index of the PDF
            converter: Converter shared by all the PDFs of the model
        """
        if pdf.is_kde():
            PDFBuilder.build_signal_kde(
                pdf,
                name,
                ipdf
            )
            return
        if pdf.is_hist():
            PDFBuilder.build_signal_hist(
                pdf,
                obs,
                name,
                ipdf
            )
            return

        config = get_signal_pdf_config(pdf.kind)
        zfit_pars = PDFBuilder._convert_parameters(pdf, converter)

        mapping = config.get('args_mapping', {})  # zfit argument -> flarefly parameter name
        pdf.pdf = config['pdf_class'](
            obs=obs, **{arg: zfit_pars[mapping.get(arg, arg)] for arg in config['pdf_args']}
        )

        if pdf.at_threshold:
            signalthr_pdf = cpdf.Pow(
                obs=obs,
                mass=zfit_pars['massthr'],
                power=zfit_pars['powerthr']
            )
            pdf.pdf = zfit.pdf.ProductPDF([pdf.pdf, signalthr_pdf], obs=obs)

    @staticmethod
    def build_signal_kde(
        pdf: F2PDFBase,
        name: str,
        ipdf: int,
    ):
        """Build a signal KDE PDF.

        Args:
            pdf: The PDF to build
            name: Base name for the PDF
            ipdf: Index of the PDF
        """
        if not pdf.kde_sample:
            Logger(f'Missing datasample for Kernel Density Estimation of signal {ipdf}!', 'FATAL')

        kde_options = pdf.kde_option or {}

        pdf.pdf = get_kde_pdf(pdf.kind)(
            data=pdf.kde_sample.get_data(),
            obs=pdf.kde_sample.get_obs(),
            name=f'{name}_kde_signal{ipdf}',
            **kde_options
        )

    @staticmethod
    def build_signal_hist(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
    ):
        """Build a signal PDF from a histogram template.

        Args:
            pdf: The PDF to build
            obs: The observable space for the PDF
            name: Base name for the PDF
            ipdf: Index of the PDF
        """
        if not pdf.hist_sample:
            Logger(f'Missing datasample for histogram template of signal {ipdf}!', 'FATAL')

        pdf.pdf = zfit.pdf.SplinePDF(
            zfit.pdf.HistogramPDF(
                pdf.hist_sample.get_binned_data(),
                name=f'{name}_hist_signal{ipdf}'
            ),
            order=3,
            obs=obs
        )

    @staticmethod
    def build_bkg_pdf(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
        converter: ZfitParameterConverter,
    ):
        """Build a background PDF with configurable parameters.

        Args:
            pdf: The PDF to build
            obs: The observable space for the PDF
            name: Base name for parameters
            ipdf: Index of the PDF
            converter: Converter shared by all the PDFs of the model
        """
        if pdf.is_kde():
            PDFBuilder.build_bkg_kde(pdf, name, ipdf)
            return
        if pdf.is_hist():
            PDFBuilder.build_bkg_hist(pdf, obs, name, ipdf)
            return
        if pdf.kind == PDFType.CHEBPOL:
            # Handle Chebyshev polynomials specially
            PDFBuilder._build_chebyshev_pdf(pdf, obs, name, ipdf, converter)
            return

        config = get_bkg_pdf_config(pdf.kind)
        zfit_pars = PDFBuilder._convert_parameters(pdf, converter)

        mapping = config.get('args_mapping', {})  # zfit argument -> flarefly parameter name
        pdf.pdf = config['pdf_class'](
            obs=obs, **{arg: zfit_pars[mapping.get(arg, arg)] for arg in config['pdf_args']}
        )

    @staticmethod
    def _build_chebyshev_pdf(
        pdf: F2PDFBase,
        obs: zfit.Space,
        converter: ZfitParameterConverter,
    ):
        """Build a Chebyshev polynomial background PDF."""
        zfit_pars = PDFBuilder._convert_parameters(pdf, converter)

        pdf.pdf = zfit.pdf.Chebyshev(
            obs=obs,
            coeff0=zfit_pars['c0'],
            coeffs=[zfit_pars[f'c{deg}'] for deg in range(1, pdf.kind.order + 1)]
        )

    @staticmethod
    def build_bkg_kde(
        pdf: F2PDFBase,
        name: str,
        ipdf: int,
    ):
        """Build a background KDE PDF.

        Args:
            pdf: The PDF to build (must have kde_sample and kde_option set)
            name: Base name for the PDF
            ipdf: Index of the PDF
        """
        if not pdf.kde_sample:
            Logger(f'Missing datasample for Kernel Density Estimation of background {ipdf}!', 'FATAL')

        kde_options = pdf.kde_option or {}

        pdf.pdf = get_kde_pdf(pdf.kind)(
            data=pdf.kde_sample.get_data(),
            obs=pdf.kde_sample.get_obs(),
            name=f'{name}_kde_bkg{ipdf}',
            **kde_options
        )

    @staticmethod
    def build_bkg_hist(
        pdf: F2PDFBase,
        obs: zfit.Space,
        name: str,
        ipdf: int,
    ):
        """Build a bkg PDF from a histogram template.

        Args:
            pdf: The PDF to build
            obs: The observable space for the PDF
            name: Base name for the PDF
            ipdf: Index of the PDF
        """
        if not pdf.hist_sample:
            Logger(f'Missing datasample for histogram template of background {ipdf}!', 'FATAL')

        pdf.pdf = zfit.pdf.SplinePDF(
            zfit.pdf.HistogramPDF(
                pdf.hist_sample.get_binned_data(),
                name=f'{name}_hist_bkg{ipdf}'
            ),
            order=3,
            obs=obs
        )

    @staticmethod
    def _convert_parameters(
        pdf: F2PDFBase,
        converter: ZfitParameterConverter,
    ):
        """Convert the shape parameters of a PDF to zfit parameters.

        Args:
            pdf: The PDF whose parameters are converted
            converter: Converter shared by all the PDFs of the model

        Returns:
            Dictionary {short name: zfit parameter}
        """
        zfit_pars = {}
        for par_name, par in pdf.parameters.items():
            if par_name == 'frac':  # fractions are built by the composer
                continue
            zfit_pars[par_name] = converter.convert(par)
        return zfit_pars
