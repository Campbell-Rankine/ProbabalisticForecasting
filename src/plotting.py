import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import os


def gen_plot(
    outputs, batch, epoch, figheight: Optional[int] = 5, figwidth: Optional[int] = 7
):

    path = "./testing/results"

    for idx, row in enumerate(batch["past_values"]):

        # build plot
        fig, ax = plt.subplots()
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)

        # get correct save_loc
        save_loc = path + f"/epoch-{epoch}"
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)

        save_loc = save_loc + f"/stock-{idx}-epoch-{epoch}.png"

        prediction_length = outputs[idx].shape[1]

        ax.plot(list(range(30)), row[-30:], label="past close", color="blue")
        ax.plot(
            list(range(30, 30 + prediction_length)),
            batch["future_values"][idx],
            label="future close",
            color="red",
        )

        plt.plot(
            list(range(30, 30 + prediction_length)),
            np.median(outputs[idx], axis=0),
            label="Prediction median",
            color="green",
            alpha=0.3,
        )

        plt.plot(
            list(range(30, 30 + prediction_length)),
            np.mean(outputs[idx], axis=0),
            label="Prediction mean",
            color="purple",
            alpha=0.3,
        )

        plt.fill_between(
            list(range(30, 30 + prediction_length)),
            outputs[idx].mean(0) - (0.5 * outputs[idx].std(axis=0)),
            outputs[idx].mean(0) + (0.5 * outputs[idx].std(axis=0)),
            color="orange",
            alpha=0.3,
            interpolate=True,
            label="+/- 1-std",
        )

        plt.legend()
        plt.savefig(save_loc)
        plt.close()
