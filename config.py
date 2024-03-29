from gluonts.time_feature import get_lags_for_frequency

model_config = {
    "prediction_length": 4,
    "context_length": 90,
    "lags_sequence": get_lags_for_frequency("1D"),
    "num_time_features": 5,  # open, high, low, close, volume
    "num_static_categorical_features": 1,  # ticker
    "cardinality": -1,  # Set this on input
    "embedding_dimension": [2],
    # Transformer params
    "encoder_layers": 4,
    "decoder_layers": 4,
    "d_model": 32,
}
