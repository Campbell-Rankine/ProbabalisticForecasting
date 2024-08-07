"""
Pull the last n-years of stock data
"""

import os
import yahoo_fin.stock_info as yf
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
import pandas as pd
from datetime import datetime
from timeit import default_timer as timer
import pickle
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from typing import Optional

# logging
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


# Globals
workers = multiprocessing.cpu_count()
date = datetime.now().strftime("%Y-%m-%d")
one_year = (
    str(int(date.split("-")[0]) - 2)
    + "-"
    + date.split("-")[1]
    + "-"
    + str(int(date.split("-")[2]) - 2)
)

save_loc = "./train/train_data.pickle"
test_save_loc = "./testing/test_data.pickle"


def pull_ticker_data(ticker):
    try:
        data = yf.get_data(ticker=ticker, start_date=one_year)
        data = data.drop(["ticker"], axis=1)

        logging.info(f"Downloading: {ticker}, Price Variance: {np.std(data['close'])}")
        if np.std(data["close"]) <= 0.7:
            return None

        data["mu"] = np.mean(data["open"]) * np.ones_like(data["open"].to_numpy())
        data["sigma"] = np.std(data["open"]) * np.ones_like(data["open"].to_numpy())
        data = data.to_dict(orient="list")
        return (ticker, data)
    except Exception as e:
        logging.info(f"Unable to download: {ticker}, Reason: {e}")
        return None


def calculate_target_variable(data: dict, n: Optional[int] = -1):
    """
    Python generator to built the target variable from some df
    Daily Returns = (CLOSE_T / CLOSE_(T-1)) - 1

    Args:
        data: pd.DataFrame - OHLC df for stock_i
        n: int - data.shape[0] if converting to list. If operating over large or continuous dfs use -1
    """
    assert n <= len(data["close"]) and "close" in data.keys()
    iterations: int = 0

    if iterations == 0:
        yield 0
        prev_value: float = data["close"][0]
        curr_value: float = data["close"][1]

    match n:
        case -1:
            yield (curr_value / prev_value) - 1
            prev_value = curr_value
            iterations += 1
            curr_value = data["close"][iterations]
        case _:
            while iterations < n - 1:
                yield (curr_value / prev_value) - 1
                prev_value = curr_value
                iterations += 1
                curr_value = data["close"][iterations]


def flatten_stocks(data: dict[pd.DataFrame], limiter: Optional[int] = -1):
    """
    Flatten pricing data and add ID as categorical variable for time series data processing
    """
    # initialize return obect
    return_dict = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "target": [],
        "adjclose": [],
        "volume": [],
        "feat_static_cat": [],
        "start": [],
        "mu": [],
        "sigma": [],
        "exponential-avg": [],  # List storing Exponential Moving Average
        "delta-ema": [],  # List storing (TARGET-EMA)
    }

    # iterate over stocks
    databar = tqdm(enumerate(data.items()))
    for idx, (stock, data_) in databar:
        databar.set_description(f"Building final return dictionary for: {stock}")
        return_dict["open"].extend(data_["open"])
        return_dict["high"].extend(data_["high"])
        return_dict["low"].extend(data_["low"])
        return_dict["close"].extend(
            data_["close"]
        )  # TODO: Target, daily return (tomorrow and today's close). Extra feature. T_1 - T_N (not T_0)
        return_dict["adjclose"].extend(data_["adjclose"])
        return_dict["volume"].extend(data_["volume"])
        return_dict["mu"].extend(data_["mu"])
        return_dict["sigma"].extend(data_["sigma"])

        id = []
        start = []
        for _ in range(len(data_["open"])):
            id.append(idx)
            start.append(one_year)

        return_dict["feat_static_cat"].extend(id)
        return_dict["start"].extend(start)
        del id
        del start

        # Calculate three day moving average
        ema = pd.DataFrame(data_["close"]).ewm(span=7).mean()
        ema = ema[ema.columns[0]].to_list()

        logging.info(f"\nShape Comparison. EMA: {len(ema)}, DF: {len(data_['close'])}")
        assert len(ema) == len(data_["close"])

        # Add additional features here:
        return_dict["exponential-avg"].extend(ema)
        return_dict["delta-ema"].extend(list(np.array(data_["close"]) - np.array(ema)))
        return_dict["target"].extend(
            list(calculate_target_variable(data=data_, n=len(data_["close"])))
        )

        assert len(return_dict["target"]) == len(return_dict["close"])

    # add dim
    for key, val in return_dict.items():

        return_dict[key] = np.nan_to_num(val, 0.0)
        return_dict[key] = return_dict[key].reshape(1, -1)

    return return_dict


if __name__ == "__main__":
    print("Running download script on %i cores" % workers)
    print(one_year)

    # Alternate these two datasets epoch to epoch
    tickers = yf.tickers_nasdaq()
    # tickers = yf.tickers_ftse250()
    # tickers.extend(yf.tickers_sp500())
    # tickers.extend(yd.tickers_dow())

    inds = np.random.uniform(
        0.0, len(tickers), len(tickers)
    )  # in the case where param 3 = len(tickers), get random shuffle of indices for nasdaq tickers
    inds = [int(x) for x in inds]
    tickers = [tickers[ind] for ind in inds]  # Debug flag application

    print(f"Downloading {len(tickers)} tickers from date: {one_year}")

    download_start = timer()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(pull_ticker_data, tickers)

    list_dfs = list(futures)
    list_dfs = list(filter(lambda item: item is not None, list_dfs))

    # Train test inf splitting
    train = list_dfs[: len(list_dfs) - 300]
    test = list_dfs[len(list_dfs) - 300 :]

    download_end = timer()
    print(f"Downloading ticker data took: {round(download_end - download_start, 2)} s")

    dict_start = timer()
    train_save = {}
    for return_val in train:
        if not return_val is None:
            train_save[return_val[0]] = return_val[1]

    dict_end = timer()
    print(f"Saving {len(list(train_save.keys()))} tickers")

    dict_start = timer()
    test_save = {}
    for return_val in test:
        if not return_val is None:
            test_save[return_val[0]] = return_val[1]

    dict_end = timer()
    print(f"Saving {len(list(test_save.keys()))} tickers")

    print("Saving individual binaries for data investigation/processing")
    with open(save_loc, "wb") as f:
        pickle.dump(train_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(test_save_loc, "wb") as f:
        pickle.dump(test_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Building flattened train dict")

    train_dict = flatten_stocks(train_save)
    train = Dataset.from_dict(train_dict)

    test_dict = flatten_stocks(test_save)
    test = Dataset.from_dict(test_dict)

    save = DatasetDict({"train": train, "test": test})
    save.save_to_disk("./data")

    print("FINISHED")
