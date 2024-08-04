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
from model_helpers import load_model_parameters

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f"Imports finished. Start run.")

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

    features = list(train[0].keys())
    cardinality = len(train[0]["open"])
    batches_per_epoch = int(cardinality / args.batch)

    data_params = {
        "prediction_length": args.pred,
        "context_length": args.context,
        "freq": args.freq,
        "categorical": 1,
        "cardinality": len(train[0]["open"]),
        "dynamic_real": 8,
    }

    model_params = {
        "embedding_dim": 2,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "d_model": 512,
    }

    if args.verbose:
        logging.info(
            f"Dataset cardinality: {cardinality} items, Batches per Epoch: {batches_per_epoch}"
        )
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

    # handle retrain case
    path_to_weights = args.retrain
    checkpoint_dict = None
    if not path_to_weights == "":
        logging.info(f"Loading model weights from checkpoint path: {path_to_weights}")
        logging.info(f"Feature list:\n---------------------\n {features}")
        checkpoint_dict = load_model_parameters(path_to_weights)
        logging.info(checkpoint_dict.keys())
        logging.info(f"Resuming training run from epoch: {checkpoint_dict['epoch']+1}")
        model.load_from_weight_file(checkpoint_dict["model_state_dict"])

    # create dataloaders
    print("Creating data loaders")
    train_dl = create_train_dataloader(
        config=model.config,
        freq=args.freq,
        data=train,
        batch_size=args.batch,
        num_batches_per_epoch=batches_per_epoch,  # run through each
    )

    test_dl = create_train_dataloader(
        config=model.config,
        freq=args.freq,
        data=test,
        batch_size=args.batch,
        num_batches_per_epoch=10,  # consider this NUM_RETRIES
    )

    print("")

    if args.verbose:
        if not checkpoint_dict is None:
            train_model(
                model,
                train_dl,
                test_dl,
                use_tb=True,
                logger=logging.getLogger(),
                batch_size=args.batch,
                use_test=True,
                num_batches_per_epoch=batches_per_epoch,
                epochs=(int(args.epochs) - (checkpoint_dict["epoch"] + 1)),
                checkpoint_dict=checkpoint_dict,
                retest=args.retest,
            )
        else:
            train_model(
                model,
                train_dl,
                test_dl,
                use_tb=True,
                logger=logging.getLogger(),
                batch_size=args.batch,
                use_test=True,
                num_batches_per_epoch=batches_per_epoch,
                epochs=int(args.epochs),
                retest=args.retest,
            )
    else:
        train_model(model, train_dl, test_dl, use_tb=True, batch_size=args.batch)


if __name__ == "__main__":
    args = parse_cli()
    main(args)
