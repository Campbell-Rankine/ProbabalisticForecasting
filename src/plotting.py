import matplotlib.pyplot as plt
from typing import Optional
import numpy as np


def gen_plot(
    outputs, batch, epoch, figheight: Optional[int] = 5, figwidth: Optional[int] = 7
):

    path = "./test/results"
    for idx in range(outputs.shape[0]):
        fig, ax = plt.subplots()
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)

        past = batch["past_values"][idx][-(42 + 1) :]
        future = batch["future_values"][idx]

        # plot target variable
        ax.plot(list(range(42 + 1)), past, label="past close", color="blue")
        ax.plot(list(range(42, 42 + 21)), future, label="future close", color="red")

        # plot prediction values
        plt.plot(
            list(range(42, 42 + 21)),
            np.median(outputs[idx], axis=0),
            label="Prediction median",
            color="green",
            alpha=0.3,
        )

        plt.fill_between(
            list(range(42, 42 + 21)),
            outputs[idx].mean(0) - outputs[idx].std(axis=0),
            outputs[idx].mean(0) + outputs[idx].std(axis=0),
            color="orange",
            alpha=0.3,
            interpolate=True,
            label="+/- 1-std",
        )

        plt.legend()
        plt.savefig(f"{path}/batch_{idx}_epoch_{epoch}.png")
