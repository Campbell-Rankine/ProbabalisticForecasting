"""
Pull the last year of stock data
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
    str(int(date.split("-")[0]) - 1)
    + "-"
    + date.split("-")[1]
    + "-"
    + str(int(date.split("-")[2]) - 2)
)

save_loc = "./train/train_data.pickle"
test_save_loc = "./test/test_data.pickle"


def pull_ticker_data(ticker):
    try:
        data = yf.get_data(ticker=ticker, start_date=one_year)
        data = data.drop(["ticker"], axis=1)

        logging.info(f"Downloading: {ticker}, Price Variance: {np.std(data['close'])}")
        if np.std(data["close"]) <= 0.5:
            return None

        data["mu"] = np.mean(data["open"]) * np.ones_like(data["open"].to_numpy())
        data["sigma"] = np.std(data["open"]) * np.ones_like(data["open"].to_numpy())
        data = data.to_dict(orient="list")
        return (ticker, data)
    except Exception as e:
        logging.info(f"Unable to download: {ticker}, Reason: {e}")
        return None


def flatten_stocks(data: dict, limiter: Optional[int] = -1):
    """
    Flatten pricing data and add ID as categorical variable for time series data processing
    """
    # initialize return obect
    return_dict = {
        "open": [],
        "high": [],
        "low": [],
        "target": [],
        "adjclose": [],
        "volume": [],
        "feat_static_cat": [],
        "start": [],
        "mu": [],
        "sigma": [],
    }

    # iterate over stocks
    databar = tqdm(enumerate(data.items()))
    for idx, (stock, data_) in databar:
        databar.set_description(f"Building final return dictionary for: {stock}")
        return_dict["open"].extend(data_["open"])
        return_dict["high"].extend(data_["high"])
        return_dict["low"].extend(data_["low"])
        return_dict["target"].extend(data_["close"])
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

    # add dim
    for key, val in return_dict.items():

        return_dict[key] = np.nan_to_num(val, 0.0)
        return_dict[key] = return_dict[key].reshape(1, -1)
    return return_dict


if __name__ == "__main__":
    print("Running download script on %i cores" % workers)
    print(one_year)

    tickers = yf.tickers_nasdaq()
    inds = np.random.uniform(0.0, len(tickers), 650)
    inds = [int(x) for x in inds]
    tickers = [tickers[ind] for ind in inds]  # Debug flag application

    print(f"Downloading {len(tickers)} tickers from date: {one_year}")

    download_start = timer()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(pull_ticker_data, tickers)

    list_dfs = list(futures)
    list_dfs = list(filter(lambda item: item is not None, list_dfs))

    # Train test inf splitting
    train = list_dfs[:450]
    test = list_dfs[450:]

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
