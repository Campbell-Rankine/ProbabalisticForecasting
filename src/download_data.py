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
    + str(int(date.split("-")[2]) - 1)
)

save_loc = "./price_predictions/train/train_data.pickle"
test_save_loc = "./price_predictions/test/test_data.pickle"


def pull_ticker_data(ticker):
    try:
        data = yf.get_data(ticker=ticker, start_date=one_year)
        data = data.drop(["ticker"], axis=1)

        logging.info(f"Downloading: {ticker}, Price Variance: {np.std(data['close'])}")
        if np.std(data["close"]) == 0.0:
            return None

        data = data.to_dict(orient="records")
        return (ticker, data)
    except Exception as e:
        logging.info(f"Unable to download: {ticker}, Reason: {e}")
        return None


if __name__ == "__main__":
    print("Running download script on %i cores" % workers)
    print(one_year)

    tickers = yf.tickers_nasdaq()
    inds = np.random.uniform(0.0, len(tickers), 675)
    inds = [int(x) for x in inds]
    tickers = [tickers[ind] for ind in inds]  # Debug flag application

    print(f"Downloading {len(tickers)} tickers")

    download_start = timer()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(pull_ticker_data, tickers)

    list_dfs = list(futures)
    list_dfs = list(filter(lambda item: item is not None, list_dfs))

    # Train test inf splitting
    train = list_dfs[:500]
    test = list_dfs[500:600]

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

    with open(save_loc, "wb") as f:
        pickle.dump(train_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(test_save_loc, "wb") as f:
        pickle.dump(test_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("FINISHED")

"""
TODO:
  1) Modify script to download test data as well
"""
