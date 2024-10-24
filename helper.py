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
        default=92,
        type=int,
        help="Override context length",
    )

    parser.add_argument(
        "-batch",
        "--batch",
        dest="batch",
        metavar="batch",
        default=24,
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

    parser.add_argument(
        "-retest",
        "--retest",
        dest="retest",
        metavar="retest",
        default=False,
        type=bool,
        help="Toggle training run restart testing inference (default : False)",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        metavar="epochs",
        default=70,
        type=int,
        help="Num training epochs (default : 70)",
    )

    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        metavar="debug",
        default=False,
        type=bool,  # TODO: Implement functionality to load after iterations.
        help="Debug mode flag. (default : True)",  # TODO: Write code that saves model to checkpoint: Forecaster-checkpoint-{epoch}.pth
    )

    parser.add_argument(
        "-pen",
        "--penalty",
        dest="penalty",
        metavar="penalty",
        default="None",
        type=str,
        help="Debug mode flag. (default : 'None')",
    )

    parser.add_argument(
        "-bk",
        "--backprop",
        dest="backprop",
        metavar="backprop",
        default=1,
        type=int,
        help="Skip backprop for k-iters. (default : 1(don't skip))",
    )

    parser.add_argument(
        "-a",
        "--accelerator",
        dest="acc",
        metavar="device",
        default="gpu",
        type=str,
        help="Lightning device. (default : 1(don't skip))",
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
