import torch as T
from ProbabalisticForecaster import ProbForecaster
from src.utils import load_data

df_list = load_data()

data_params = {
    "prediction_length": 7,
    "context_length": 30,
    "freq": "1D",
    "features": int,
    "categorical": int,
    "cardinality": int,
}
