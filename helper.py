import argparse
from typing import Optional
import logging


def parse_cli() -> argparse.Namespace:
    # Parse command line arguments for script
    parser = argparse.ArgumentParser()

    ### - Global Params - ###
    parser.add_argument(
        "-freq",
        "--freq",
        dest="freq",
        metavar="freq",
        default="1D",
        type=str,
        help="Override frequency string",
    )

    parser.add_argument(
        "-verbose",
        "--verbose",
        dest="verbose",
        metavar="verbose",
        default=True,
        type=bool,
        help="Toggle logging",
    )

    parser.add_argument(
        "-pred",
        "--pred",
        dest="pred",
        metavar="pred",
        default=14,
        type=int,
        help="Override prediction length",
    )

    parser.add_argument(
        "-con",
        "--context",
        dest="context",
        metavar="context",
        default=182,
        type=int,
        help="Override context length",
    )

    parser.add_argument(
        "-batch",
        "--batch",
        dest="batch",
        metavar="batch",
        default=7,
        type=int,
        help="Override context length",
    )

    parser.add_argument(
        "-test",
        "--test",
        dest="test",
        metavar="test",
        default=True,
        type=bool,
        help="Toggle testing inference",
    )

    parser.add_argument(
        "-retrain",
        "--retrain",
        dest="retrain",
        metavar="retrain",
        default="",
        type=str,
        help="Toggle testing inference",
    )

    args = parser.parse_args()
    return args


def args_handler(fn: callable, data: dict):
    try:
        fn(**data)
    except Exception as e:
        raise ValueError(f"Unable to verify data arguments: {e}")


def logging_handler(
    verbose: bool, message: str, logger: Optional[logging.Logger] = None
):
    """
    handle logging requirements for the forecaster
    """
    if verbose and (not logger is None):
        logger.info(f"Forecaster Logging: {message}")
