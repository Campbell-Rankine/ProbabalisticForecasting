import pickle
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache


@lru_cache(10_000)
def convert_date_to_period(date, freq):
    return pd.Period(date, freq)


def load_data(train=True):
    """
    Load binary datasets into their dictionary
    Args:
      train (bool) - load training dataset toggle (default: True)
    """
    if train:
        loc = "./price_predictions/train/train_data.pickle"
    else:
        loc = "./price_predictions/test/test_data.pickle"

    with open(loc, "rb") as f:
        data = pickle.load(f)
        return data


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
