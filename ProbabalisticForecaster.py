import torch as T
import torch.nn as nn

from gluonts.time_feature import get_lags_for_frequency

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


class ProbForecaster(nn.Module):
    """
    Wrapper module for the Time Series forecaster that'll take the data you need and automatically create and return the model object.
    Also contains a list of property/logging requirements.
    """

    def __init__(
        self,
        data_params: dict,
        transformer_params: dict,
        verbose=False,
    ):
        # copy params
        self.data_params = data_params
        self.transformer_params = transformer_params

        # load data parameters
        try:
            self.verify_data_args(**data_params)
        except Exception as e:
            raise ValueError(f"Unable to verify data arguments: {e}")

        # load model params
        try:
            self.verify_model_params(**transformer_params)
        except Exception as e:
            raise ValueError(f"Unable to verify data arguments: {e}")

        self.create_model_config_object()

        self.transformer = TimeSeriesTransformerForPrediction(self.config)

    def create_model_config_object(self):
        # Load into config
        self.config = TimeSeriesTransformerConfig(
            prediction_length=self.prediction_length,
            # context length:
            context_length=self.context_length,
            # lags coming from helper given the freq:
            lags_sequence=self.lags,
            # we'll add 2 time features ("month of year" and "age", see further):
            num_time_features=len(self.features) + 1,
            # we have a single static categorical feature, namely time series ID:
            num_static_categorical_features=1,
            # it has 366 possible values:
            cardinality=[len(self.cardinality)],
            # the model will learn an embedding of size 2 for each of the 366 possible values:
            embedding_dimension=[2],
            # transformer params:
            encoder_layers=4,
            decoder_layers=4,
            d_model=32,
        )

    def verify_data_args(
        self,
        prediction_length: int,
        context_length: int,
        freq: str,
        features: int,
        categorical: int,
        cardinality: int,
    ):
        # verify data
        assert (
            type(freq) == str
            and prediction_length > 0
            and context_length > 0
            and features > 0
            and categorical > 0
            and cardinality > 0
        )
        # copy data
        self.prefiction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.features = features
        self.categorical = categorical
        self.cardinality = cardinality

        self.lags = get_lags_for_frequency(self.freq)

    def verify_model_params(
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
