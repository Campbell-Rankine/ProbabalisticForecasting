import pandas as pd
import numpy as np
from typing import Optional
import logging
from tqdm import tqdm
import os

import torch as T
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from src.utils import load_data, convert_date_to_period, send_args_to_device
from ProbabalisticForecaster import ProbForecaster
from config import model_config, hyperparams
from model_helpers import save_model_params
from src.plotting import gen_plot


# globals
accelerator = Accelerator()
device = accelerator.device


def logging_handler(message: str, logger: Optional[logging.Logger] = None):
    """
    handle logging requirements for the forecaster
    """
    if not logger is None:
        logger.info(f"Forecaster Logging: {message}")


def train_model(
    model: ProbForecaster,
    train_dl,
    test_dl,
    use_tb: Optional[bool] = False,
    use_test: Optional[bool] = False,
    logger: Optional[logging.Logger] = None,
    epochs: Optional[int] = 301,
    batch_size: Optional[int] = 64,
    num_batches_per_epoch: Optional[
        int
    ] = 140,  # also set in main but if you change this just pass it to the
    logging_path: Optional[str] = "./logging",
):
    message = f"Initializing model training conditions \n --------------------------- \n Using Tensorboard: {use_tb} \n Using Test Data: {use_test} \n Learning Rate: {hyperparams['lr']} \n Weight Decay: {hyperparams['weight_decay']} \n Betas: {hyperparams['betas']} \n Device: {device}"
    logging_handler(message, logger)

    if not os.path.exists(logging_path):
        logging_handler(f"Creating directory at: {logging_path}")
        os.mkdir(logging_path)

    # create optimizer
    model.to(device)
    optim = AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        betas=hyperparams["betas"],
        weight_decay=hyperparams["weight_decay"],
    )

    # use accelerator to prepare components
    model, optim, train_dl = accelerator.prepare(model, optim, train_dl)

    # lr scheduling
    sched = ExponentialLR(optim, 0.9725)

    # setup tracking / loggers for model

    # set model to train (calculate grads)
    model.train()
    databar = tqdm(range(epochs))
    tbwriter = None
    if use_tb:
        tbwriter = SummaryWriter(logging_path)

    for epoch_num in databar:
        losses = []
        for idx, batch in enumerate(train_dl):
            # test batch. Comment out this code if not needed
            # for k, v in batch.items():
            #     print(k, v.shape, v.type())
            # assert False

            # zero the optim gradients
            optim.zero_grad()

            # normalize the input according to xi-median / IQR ()
            q75, q25 = np.percentile(batch["past_values"].cpu().numpy(), [75, 25])
            iqr = q75 - q25

            batch["past_values"] = (
                batch["past_values"] - T.median(batch["past_values"])
            ) / iqr

            # send batch to device
            args = {
                "past_values": batch["past_values"].to(device),
                "past_time_features": batch["past_time_features"].to(device),
                "past_observed_mask": batch["past_observed_mask"].to(device),
                "static_categorical_features": batch["static_categorical_features"].to(
                    device
                ),
                "future_values": batch["future_values"].to(device),
                "future_time_features": batch["future_time_features"].to(device),
                "future_observed_mask": batch["future_observed_mask"].to(device),
                "output_hidden_states": True,
            }

            outputs = model(args, idx, batch_size=batch_size)
            loss = outputs.loss

            losses.append(loss.item())

            databar.set_description(
                f"Epoch: {epoch_num}, Iteration: {idx} / {num_batches_per_epoch} | {round(100*(idx/num_batches_per_epoch), 2)}%, Loss: {loss}, IQR: {iqr}"
            )

            # backprop
            accelerator.backward(loss)
            optim.step()

            # Write epoch data to tensorboard
            if epoch_num % 5 == 0 and not tbwriter is None:
                grad_norm = model.get_grad_norm()
                tbwriter.add_scalars(
                    f"Epoch {epoch_num} Train",
                    {
                        "Loss": loss,
                        "Current Mean Loss": np.mean(losses),
                    },
                    idx,
                )
                tbwriter.add_scalar("Gradient Norm", grad_norm, (epoch_num + 1) * idx)

        # Step LR scheduler
        sched.step()

        # model to eval
        if use_test and epoch_num % 10 == 0:
            model.eval()
            optim.zero_grad()

            test_losses = []
            last_test_loss = 0.0
            for idx__, batch in enumerate(test_dl):
                stddev_test = np.std(batch["past_values"].numpy()[:-60])
                if stddev_test < 0.1:
                    print(
                        f"Skipping print as past_values only have variance of: {stddev_test}"
                    )
                    continue

                q75, q25 = np.percentile(batch["past_values"].cpu().numpy(), [75, 25])
                iqr = q75 - q25

                past_vals = (
                    batch["past_values"] - T.median(batch["past_values"])
                ) / iqr
                args = {
                    "past_values": past_vals.to(device),
                    "past_time_features": batch["past_time_features"].to(device),
                    "past_observed_mask": batch["past_observed_mask"].to(device),
                    "static_categorical_features": batch[
                        "static_categorical_features"
                    ].to(device),
                    "future_time_features": batch["future_time_features"].to(device),
                    "output_hidden_states": True,
                }
                outputs = model(args, idx__, batch_size=batch_size, test=True)

                output = outputs.sequences.cpu().numpy()

                gen_plot(output, batch, epoch_num)
                break

            model.train()

    # shutoff
    tbwriter.close()
    save_model_params(
        model=model, optimizer=optim, losses=losses, epoch=epoch_num, scheduler=sched
    )
