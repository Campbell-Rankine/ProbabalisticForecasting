"""
Run test script
"""

import torch as T
from torch.optim import AdamW
from accelerate import Accelerator
import numpy as np
from typing import Optional
import logging

from ProbabalisticForecaster import ProbForecaster
from src.plotting import gen_plot


def test_fn(
    test_dl,
    device: str,
    batch_size: int,
    epoch_num: int,
    model: ProbForecaster,
    optim: AdamW,
    accelerator: Accelerator,
    use_test: Optional[bool] = True,
    save_every: Optional[int] = 1,
    logger: Optional[logging.Logger] = None,
):
    if use_test == False:
        return False
    # model to eval
    if use_test and epoch_num % (save_every) == 0:
        optim.zero_grad()

        test_losses = []
        last_test_loss = 0.0
        for idx__, batch in enumerate(test_dl):
            logging.info(f"Processing testing batch: {idx__}")
            stddev_test = np.std(batch["past_values"].numpy()[:-60])
            if stddev_test < 0.1:
                print(
                    f"Skipping print as past_values only have variance of: {stddev_test}"
                )
                continue

            q75, q25 = np.percentile(batch["past_values"].cpu().numpy(), [75, 25])
            iqr = q75 - q25

            past_vals = (batch["past_values"] - T.median(batch["past_values"])) / iqr

            args = {
                "past_values": past_vals.to(device),
                "past_time_features": batch["past_time_features"].to(device),
                "past_observed_mask": batch["past_observed_mask"].to(device),
                "static_categorical_features": batch["static_categorical_features"].to(
                    device
                ),
                "future_values": batch["future_values"].to(device),
                "future_time_features": batch["future_time_features"].to(device),
                "future_observed_mask": batch["future_observed_mask"].to(device),
                "output_hidden_states": False,
            }
            try:
                T.cuda.empty_cache()

                outputs = model(args, idx__, batch_size=batch_size, test=True)
                output = outputs.sequences.cpu().numpy()

                gen_plot(output, batch, epoch_num)
                break
            except Exception as e:
                logger.warning(f"Encountered error in test forward call: {e}")
                raise e

    model.train()
    return True
