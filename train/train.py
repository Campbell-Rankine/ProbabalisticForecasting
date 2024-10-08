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
    debug: Optional[bool] = False,
    reg: Optional[float] = 1e-6,
    penalty: Optional[str] = "None",
    k: Optional[int] = 1,
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
        sched = ExponentialLR(optim, 0.9975)
    else:
        optim = AdamW(
            model.parameters(),
            lr=hyperparams["lr"],
            betas=hyperparams["betas"],
            weight_decay=hyperparams["weight_decay"],
        )
        sched = ExponentialLR(optim, 0.9975)
        optim.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        sched.load_state_dict(checkpoint_dict["sched_state_dict"])

    # accelerate
    model.transformer, optim, train_dl = accelerator.prepare(
        model.transformer, optim, train_dl
    )

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
    losses_sum = 0.0
    iqr_sum = 0.0
    grad_sum = 0.0
    exit = False

    # build per epoch

    for epoch_num in databar:
        # skip to current epoch if load dict not none
        if not checkpoint_dict is None:
            if epoch_num <= checkpoint_dict["epoch"]:
                continue

        losses = []
        window_loss = []
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

            if not k == 1:
                window_loss.append(loss)

            # add weight regularization penalty
            grad_penalty: float = 0.0
            match penalty:
                case "L2":
                    grad_penalty = reg * model.get_grad_norm()
                case "L1":
                    grad_penalty = reg * model.get_grad_norm(type="L1")
            loss = loss + grad_penalty

            cpu_usage, mem_usage, gpu_usage = get_resource_usage()

            databar.set_description(
                f"Epoch: {epoch_num}, Iteration: {idx} / {num_batches_per_epoch} | {round(100*(idx/num_batches_per_epoch), 2)}%, Loss: {round(loss.item(), 2)}, Rolling AVG Loss: {round(np.mean([x.item() for x in window_loss]), 2)}, IQR: {round(iqr, 2)}, Grad penalty: {round(grad_penalty, 3)}, CPU Usage: {cpu_usage}, Memory Usage: {mem_usage}, GPU Usage: {gpu_usage}"
            )

            # backprop
            if idx % k == 0 and not k == 1:
                loss_ = (1 / k) * T.sum(T.tensor(window_loss, requires_grad=True))
                accelerator.backward(loss_)
                optim.step()
                window_loss = []
            else:
                accelerator.backward(loss)
                optim.step()

            # grad norm clipping
            if penalty == "None" and idx % (3 * k) == 0:
                T.nn.utils.clip_grad_norm_(model.transformer.parameters(), max_norm=6.0)

            # Write epoch data to tensorboard
            grad_norm = None
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
            if total_iters % 10 == 0:
                if grad_norm is None:
                    grad_norm = model.get_grad_norm()
                data.append({"loss": loss.item(), "iqr": iqr, "grad": grad_norm})
                iqrs.append(iqr)
                grads.append(grad_norm)

            if debug and idx > 30:
                exit = True
                break

        # Step LR scheduler
        sched.step()

        # get statistics for epoch
        losses_sum += np.sum(np.abs(losses))  # add abs in case loss function includes -
        iqr_sum += np.sum(np.abs(iqrs))
        grad_sum += np.sum(np.abs(grads))

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
            records.send({"avg_loss": float(losses_sum / total_iters)})
            records.send({"avg_iqr": float(iqr_sum / total_iters)})
            records.send({"avg_grad_norm": float(grad_sum / total_iters)})
            records.send({"total_iters": float(total_iters)})
            try:
                records.send({"SAVE": None})
            except StopIteration:
                print("Stop Iteration flag thrown")

            # clear mem
        with open("./avg_epoch_losses.json", "w") as file:
            import json

            json.dump({str(epoch_num): np.mean(losses)}, file)

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

        if exit:
            break

    # shutoff
    try:
        records.send({"SAVE": None})
    except StopIteration:
        print("Stop Iteration flag thrown")
    try:
        _ = records.send({"STOP_CODE": None})
    except StopIteration:
        print("Stop Iteration flag thrown")

    # close script
    print("TRAINING RUN FINISHED\n\n")
    tbwriter.close()

    save_model_params(
        model=model, optimizer=optim, losses=losses, epoch=epoch_num, scheduler=sched
    )

    import sys

    sys.exit(0)
