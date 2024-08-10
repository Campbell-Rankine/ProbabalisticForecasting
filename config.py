from gluonts.time_feature import get_lags_for_frequency

model_config = {
    "prediction_length": 7,
    "context_length": 120,
    "lags_sequence": get_lags_for_frequency("1D"),
    "num_time_features": 9,  # open, high, low, close, volume
    "num_static_categorical_features": 1,  # ticker
    "cardinality": -1,  # Set this on input
    "embedding_dimension": [2],
    # Transformer params
    "encoder_layers": 10,
    "decoder_layers": 10,
    "d_model": 512,
    "output_filter": 3,
}

hyperparams = {
    "betas": (0.95, 0.99),
    "lr": 5e-4,  # Increment this down as we run off of pretrain checkpoints (currently this is for the next one)
    "weight_decay": 1e-3,
}
