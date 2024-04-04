import pandas as pd
import numpy as np
from typing import Optional
import logging
from tqdm import tqdm
import os

import torch as T
import torch.nn as nn
from torch.optim import AdamW

from accelerate import Accelerator

from src.utils import load_data, convert_date_to_period, send_args_to_device
from ProbabalisticForecaster import ProbForecaster
from config import model_config, hyperparams
from model_helpers import save_model_params

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
    use_tb: Optional[bool] = False,
    logger: Optional[logging.Logger] = None,
    epochs: Optional[int] = 50,
    batch_size: Optional[int] = 256,
):
    message = f"Initializing model training conditions \n --------------------------- \n Using Tensorboard: {use_tb} \n Learning Rate: {hyperparams['lr']} \n Weight Decay: {hyperparams['weight_decay']} \n Betas: {hyperparams['betas']}"
    logging_handler(message, logger)

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

    # setup tracking / loggers for model
    losses = []

    # set model to train (calculate grads)
    model.train()
    databar = tqdm(range(epochs))

    for epoch_num in databar:
        for idx, batch in enumerate(train_dl):
            # zero the optim gradients
            optim.zero_grad()

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
                f"Epoch: {epoch_num}, Iteration: {idx}, Loss: {loss}"
            )

            # backprop
            accelerator.backward(loss)
            optim.step()

    save_model_params(model=model, optimizer=optim, losses=losses, epoch=epoch_num)
