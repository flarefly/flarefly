"""flarefly module"""
from .fitter import F2MassFitter
from .data_handler import DataHandler

__all__ = ["F2MassFitter", "DataHandler"]


def entrypoint():
    """ This is the entrypoint: call it from command line
    """
    print("Welcome to flarefly!\n")
