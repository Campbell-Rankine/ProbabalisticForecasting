import pickle
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
from datasets import load_from_disk


@lru_cache(10_000)
def convert_date_to_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = [convert_date_to_period(date[0], freq) for date in batch["start"]]
    return batch


def load_data(path="./data"):
    # load and return dataset from the path above
    try:
        return load_from_disk("./data")
    except Exception as e:
        raise ValueError(f"Unable to load dataset from disk: {e}")


def plot_10_price(dict_dfs, keys_to_plot, figheight=8, figwidth=36):
    """
    Plot pricing history for 10 of the keys in list_dfs
    Args:
      dict_dfs
    """
    colours = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5)
    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

    for idx, axis in enumerate(axes):
        data = dict_dfs[keys_to_plot[idx]]["close"]

        axis.plot(range(len(data)), data, colours[idx])
