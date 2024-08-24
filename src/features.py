"""
Write Generators for additional features here. TODO: Aaron feature
"""

import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm

try:
    from src.utils import load_data, load_from_disk
except:
    from utils import load_data, load_from_disk


def optional_trunc(stock_a: list[float], stock_b: list[float]):
    # Truncate lists if length unequal
    if len(stock_a) == len(stock_b):
        return stock_a, stock_b
    elif len(stock_a) < len(stock_b):
        return stock_a, stock_b[: len(stock_a)]
    elif len(stock_b) < len(stock_a):
        return stock_a[: len(stock_b)], stock_b


def compare_stocks(
    A_close: list[float], B_close: list[float], n: Optional[int] = -1
) -> int:

    # truncation handling
    if not n == -1 and len(A_close) <= n and len(B_close) <= n:
        A_close = A_close[:n]
        B_close = B_close[:n]
    else:
        A_close, B_close = optional_trunc(A_close, B_close)

    # get len sequence
    n_ = len(A_close)  # guaranteed equal length at this point in execution

    # calc final return value
    close_diff = np.array(A_close) - np.array(B_close)
    sq_close_diff = close_diff**2
    mean_sq_close_spread = (1 / n_) * np.sum(sq_close_diff)

    return mean_sq_close_spread


def get_best_pair(
    stock_id,
    stock_list,
    data,
    identifier: Optional[str] = "feat_static_cat",
    comp_fn: Optional[callable] = compare_stocks,
):
    # return the best mean reversion pair for stock (stock_id: ID of stock_a)
    a_data = data[data[identifier] == stock_id]
    a_close = a_data["close"]
    stock_list_ = [x for x in stock_list if not x == stock_id]
 
    best = {"value": -1, "id": ""}

    # length check
    assert len(stock_list_) + 1 == len(stock_list)

    for comparison in stock_list_:
        comparison_data = data[data[identifier] == comparison]
        comparison_close = comparison_data["close"]

        # type conversion
        a_close = list(a_close)
        b_close = list(comparison_close)

        val = comp_fn(a_close, b_close)
        if val < best["value"]:
            # set best
            best["value"] = val
            best["id"] = comparison

    return {"stock1": stock_id, "stock2": best["id"], "val": best["value"]}


def map_stocks(
    data,
    comparison_fn: Optional[callable] = compare_stocks,
    identifier: Optional[str] = "feat_static_cat",
) -> dict:
    # Use the comparison function to compare all stocks in data
    stock_list = list(set(data[identifier]))
    print(f"Processing {len(stock_list)} stocks")

    # create databar
    databar = tqdm(enumerate(stock_list))
    n = len(stock_list)

    pairs = []
    for idx, stock in databar:
        if idx == 0:
            databar.set_description(
                f"Calculating best pair for stock with ID: {stock} | {idx}/{n} : {round(100*(idx/n), 2)}%"
            )

        result = get_best_pair(
            stock, stock_list, data, identifier=identifier, comp_fn=comparison_fn
        )
        pairs.append(result)
        databar.set_description(
            f"Calculating best pair for stock with ID: {stock}. Last pair : {result} | {idx}/{n} : {round(100*(idx/n), 2)}%"
        )

    return pairs


# TODO: Aarons feature. Just train this model and lets get this over with


if __name__ == "__main__":
    print("getting stock comparison pairs")
    train_data = load_data()["train"]
    print("Finished loading data")
    print(train_data[0]["feat_static_cat"])

    mappings = map_stocks(train_data)
