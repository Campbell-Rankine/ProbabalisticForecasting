import pandas as pd
import numpy as np
from typing import Optional
import logging
from tqdm import tqdm
import os
import psutil as _p

import torch as T
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from src.experiment_tracking import (
    experiment_data,
)  # TODO: Implement functionality into training script
from src.utils import load_data, convert_date_to_period, send_args_to_device
from ProbabalisticForecaster import ProbForecaster
from config import model_config, hyperparams
from model_helpers import save_model_params
from src.plotting import gen_plot
from testing.testing_script import test_fn


# globals
accelerator = Accelerator()
device = accelerator.device


def logging_handler(message: str, logger: Optional[logging.Logger] = None):
    """
    handle logging requirements for the forecaster
    """
    if not logger is None:
        logger.info(f"Forecaster Logging: {message}")


def get_resource_usage():
    cpu_usage = _p.cpu_percent()
    mem_usage = _p.virtual_memory().percent
    reserved, total = T.cuda.mem_get_info()
    gpu_usage = round(100.0 * float(reserved / total), 2)
    return cpu_usage, mem_usage, gpu_usage


def train_model(
    model: ProbForecaster,
    train_dl,
    test_dl,
    use_tb: Optional[bool] = False,
    use_test: Optional[bool] = False,
    logger: Optional[logging.Logger] = None,
    epochs: Optional[int] = 91,
    batch_size: Optional[int] = 64,
    num_batches_per_epoch: Optional[
        int
    ] = 2000,  # also set in main but if you change this just pass it to the dataloader
    logging_path: Optional[str] = "./logging",
    save_every: Optional[
        int
    ] = 1,  # each epoch takes about 4 hours if using the default settings
    save_path: Optional[str] = "./weights/checkpoints/",
    checkpoint_dict=None,
    retest: Optional[bool] = False,
):
    message = f"Initializing model training conditions \n --------------------------- \n Using Tensorboard: {use_tb} \n Using Test Data: {use_test} \n Learning Rate: {hyperparams['lr']} \n Weight Decay: {hyperparams['weight_decay']} \n Betas: {hyperparams['betas']} \n Device: {device}"
    logging_handler(message, logger)

    if not os.path.exists(logging_path):
        logging_handler(f"Creating directory at: {logging_path}")
        os.mkdir(logging_path)

    # create optimizer
    model.transformer.to(device)
    if checkpoint_dict is None:
        optim = AdamW(
            model.parameters(),
            lr=hyperparams["lr"],
            betas=hyperparams["betas"],
            weight_decay=hyperparams["weight_decay"],
        )
        sched = ExponentialLR(optim, 0.9725)
    else:
        optim = AdamW(
            model.parameters(),
            lr=hyperparams["lr"],
            betas=hyperparams["betas"],
            weight_decay=hyperparams["weight_decay"],
        )
        sched = ExponentialLR(optim, 0.9725)
        optim.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        sched.load_state_dict(checkpoint_dict["sched_state_dict"])

    # accelerate
    model, optim, train_dl = accelerator.prepare(model, optim, train_dl)

    # lr scheduling

    # setup tracking / loggers for model

    # set model to train (calculate grads)
    model.train()
    optim.zero_grad()

    databar = tqdm(range(epochs))
    tbwriter = None
    if use_tb:
        tbwriter = SummaryWriter(logging_path)
    epoch_num = 0
    # retest before starting
    if retest:
        _ = test_fn(
            test_dl,
            device,
            batch_size,
            epoch_num,
            model,
            optim,
            accelerator,
            use_test,
            save_every,
            logger,
        )

    # prime experiment tracking
    records = experiment_data()
    records.send(None)  # prime microservice

    data = []
    total_iters = 0
    losses_avg = 0.0
    iqr_avg = 0.0
    grad_avg = 0.0

    # build per epoch

    for epoch_num in databar:
        # skip to current epoch if load dict not none
        if not checkpoint_dict is None:
            if epoch_num <= checkpoint_dict["epoch"]:
                continue

        losses = []
        iqrs = []
        grads = []

        optim.zero_grad()

        if epoch_num % 20 == 0:
            model.transformer.zero_grad()

        for idx, batch in enumerate(train_dl):
            total_iters += 1
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

            cpu_usage, mem_usage, gpu_usage = get_resource_usage()

            databar.set_description(
                f"Epoch: {epoch_num}, Iteration: {idx} / {num_batches_per_epoch} | {round(100*(idx/num_batches_per_epoch), 2)}%, Loss: {loss}, IQR: {iqr}, CPU Usage: {cpu_usage}, Memory Usage: {mem_usage}, GPU Usage: {gpu_usage}"
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

            # save training experiment data
            data.append({"loss": loss.item(), "iqr": iqr, "grad": grad_norm})
            losses.append(loss.item())
            iqrs.append(iqr)
            grads.append(grad_norm)

        # Step LR scheduler
        sched.step()

        # get statistics for epoch
        losses_avg += np.sum(np.abs(losses))  # add abs in case loss function includes -
        iqr_avg += np.sum(np.abs(iqrs))
        grad_avg += np.sum(np.abs(grads))

        # Save model weights to checkpoint. This will get overwritten in the next save.
        if save_every > -1 and epoch_num % save_every == 0:
            logging.info(f"Saving model checkpoint for epoch: {epoch_num}")

            save_model_params(
                model=model,
                optimizer=optim,
                losses=losses,
                epoch=epoch_num,
                scheduler=sched,
                name="checkpoint",
            )

            # save experiment csv
            records.send({"data": data})
            records.send({"avg_loss": losses_avg / total_iters})
            records.send({"avg_iqr": iqr_avg / total_iters})
            records.send({"avg_grad_norm": grad_avg / total_iters})
            records.send({"total_iters": total_iters})
            records.send("SAVE")

            # clear mem

        # Handle Gradient memory for test case
        _ = test_fn(
            test_dl,
            device,
            batch_size,
            epoch_num,
            model,
            optim,
            accelerator,
            use_test,
            save_every,
            logger,
        )

    # shutoff
    records.send("SAVE")
    out = records.send("STOP_CODE")
    print("TRAINING RUN FINISHED\n\n")
    print(out)
    tbwriter.close()

    save_model_params(
        model=model, optimizer=optim, losses=losses, epoch=epoch_num, scheduler=sched
    )
