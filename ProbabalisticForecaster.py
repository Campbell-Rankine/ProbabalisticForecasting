# library imports
from datetime import datetime
import logging
from typing import Optional

# pytorch
import torch as T
import torch.nn as nn

# Higher level imports
from gluonts.time_feature import get_lags_for_frequency
from gluonts.time_feature import time_features_from_frequency_str
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


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


class ProbForecaster(nn.Module):
    """
    Wrapper module for the Time Series forecaster that'll take the data you need and automatically create and return the model object.
    Also contains a list of property/logging requirements.

    Args:
        data_params         (dict) - Unpacked and used to create TimeSeriesTransformerConfig fields. Assertions are done for verification
        transformer_params  (dict) - Unpacked and used to create TimeSeriesTransformerConfig fields. Assertions are done for verification
        verbose             (bool | None) - Logging toggle
        logger              (logging.Logger | None) - Logger, output loss etc.

    Raises:
        ValueError - Improper model config args.
    """

    def __init__(
        self,
        data_params: dict,
        transformer_params: dict,
        verbose: Optional[bool] = False,
        logger: Optional[logging.Logger] = None,
    ):
        # set up module
        super().__init__()

        # copy params
        self.data_params = data_params
        self.transformer_params = transformer_params
        self.verbose = verbose
        self.logger = logger

        if self.logger is None:
            print("Initializing with no logger, override verbose")
            self.verbose = False

        # load data parameters
        args_handler(self._verify_data_args, self.data_params)
        args_handler(self._verify_model_params, self.transformer_params)

        logging_handler(
            self.verbose, f"Finished verifying Transformer Config", self.logger
        )

        self._create_model_config_object()

        self.transformer = TimeSeriesTransformerForPrediction(self.config)

        if verbose:
            print("Model Architecture")
            print("-------------------")
            print(self.transformer)

    def _verify_data_args(
        self,
        prediction_length: int,
        context_length: int,
        freq: str,
        categorical: int,
        cardinality: int,
        dynamic_real: int,
    ):
        # verify data
        assert (
            type(freq) == str
            and prediction_length > 0
            and context_length > 0
            and categorical > 0
            and cardinality > 0
            and dynamic_real > 0
        )
        # copy data
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.categorical = categorical
        self.cardinality = cardinality
        self.dynamic_real = dynamic_real

        self.lags = get_lags_for_frequency(self.freq)
        self.time_features = time_features_from_frequency_str(self.freq)

    def _verify_model_params(
        self, embedding_dim, encoder_layers, decoder_layers, d_model
    ):
        # vet model parameters
        assert (
            embedding_dim > 0
            and encoder_layers > 0
            and decoder_layers > 0
            and d_model > 0
        )

        # load into memory
        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.d_model = d_model

    # utility functions
    def _create_model_config_object(self):
        # Load into config
        self.config = TimeSeriesTransformerConfig(
            prediction_length=self.prediction_length,
            # context length:
            context_length=self.context_length,
            # lags coming from helper given the freq:
            lags_sequence=self.lags,
            # we'll add  time feature ("month of year"):
            num_time_features=len(self.time_features) + 1,
            # we have a single static categorical feature, namely time series id:
            num_static_categorical_features=1,
            # we have multiple fields with time series data so:
            num_dynamic_real_features=self.dynamic_real,
            # Number of rows:
            cardinality=[self.cardinality],
            # the model will learn an embedding of size 2 for each of the 366 possible values:
            embedding_dimension=[self.embedding_dim],
            # transformer params:
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            d_model=self.d_model,
        )
