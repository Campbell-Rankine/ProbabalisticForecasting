import torch as T
import argparse
import logging
from functools import partial


from ProbabalisticForecaster import ProbForecaster
from src.utils import load_data, transform_start_field
from src.dataloaders import (
    create_backtest_dataloader,
    create_train_dataloader,
    create_test_dataloader,
)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

df_list = load_data()


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
        default=7,
        type=int,
        help="Override prediction length",
    )

    parser.add_argument(
        "-con",
        "--context",
        dest="context",
        metavar="context",
        default=30,
        type=int,
        help="Override context length",
    )

    parser.add_argument(
        "-batch",
        "--batch",
        dest="batch",
        metavar="batch",
        default=20,
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

    args = parser.parse_args()
    return args


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
        "cardinality": train.num_rows,
        "dynamic_real": 4,
    }

    model_params = {
        "embedding_dim": 2,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "d_model": 32,
    }

    if args.verbose:
        model = ProbForecaster(
            data_params=data_params,
            transformer_params=model_params,
            verbose=True,
            logger=logger,
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
        num_batches_per_epoch=3,
    )

    test_dl = create_test_dataloader(
        config=model.config,
        freq=args.freq,
        data=test,
        batch_size=64,
    )

    # test first batch
    batch = next(iter(train_dl))
    print(batch)


if __name__ == "__main__":
    args = parse_cli()
    main(args)
