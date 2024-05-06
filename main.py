import torch as T
import argparse
import logging
from functools import partial
import numpy as np

from ProbabalisticForecaster import ProbForecaster
from src.utils import load_data, transform_start_field
from src.dataloaders import (
    create_backtest_dataloader,
    create_train_dataloader,
    create_test_dataloader,
)
from train.train import *
from helper import parse_cli

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

df_list = load_data()

"""
Project TODO:

1) Add more features to time features. See Quantstats for the list.
"""


def main(args: argparse.Namespace):
    """
    Begin setup and context before the training / testing loops are done
    """
    # load data from disk
    data = load_data()
    train = data["train"]
    test = data["test"]

    train.set_transform(partial(transform_start_field, freq=args.freq))
    test.set_transform(partial(transform_start_field, freq=args.freq))

    data_params = {
        "prediction_length": args.pred,
        "context_length": args.context,
        "freq": args.freq,
        "categorical": 1,
        "cardinality": 1016,
        "dynamic_real": 4,
    }

    model_params = {
        "embedding_dim": 1,
        "encoder_layers": 8,
        "decoder_layers": 8,
        "d_model": 256,
    }

    if args.verbose:
        model = ProbForecaster(
            data_params=data_params,
            transformer_params=model_params,
            verbose=True,
            logger=logging.getLogger(),
        )
    else:
        model = ProbForecaster(
            data_params=data_params,
            transformer_params=model_params,
            verbose=False,
        )

    # create dataloaders
    print("Creating data loaders")
    train_dl = create_train_dataloader(
        config=model.config,
        freq=args.freq,
        data=train,
        batch_size=args.batch,
        num_batches_per_epoch=144,  # run through each
    )

    test_dl = create_train_dataloader(
        config=model.config,
        freq=args.freq,
        data=test,
        batch_size=args.batch,
        num_batches_per_epoch=1,
    )

    print("")

    if args.verbose:
        train_model(
            model,
            train_dl,
            test_dl,
            use_tb=True,
            logger=logging.getLogger(),
            batch_size=args.batch,
            use_test=True,
            num_batches_per_epoch=144,
        )
    else:
        train_model(model, train_dl, test_dl, use_tb=True, batch_size=args.batch)


if __name__ == "__main__":
    args = parse_cli()
    main(args)
