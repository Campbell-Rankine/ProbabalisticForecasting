import torch as T
from datetime import datetime
import os
from typing import Optional

from ProbabalisticForecaster import ProbForecaster

save_loc = "./weights"  # change if need be
if not os.path.exists(save_loc):
    os.mkdir(save_loc)

today = datetime.now().strftime("%Y-%m-%d")


def save_model_params(
    model: ProbForecaster,
    optimizer: T.optim.AdamW,
    losses: list,
    epoch: int,
    scheduler: T.optim.lr_scheduler.ExponentialLR,
    name: Optional[str] = None,
):
    """
    Save full training run including losses, optimizer etc.
    """
    # create save dict
    to_save = {
        "epoch": epoch,
        "loss": losses,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_state_dict": model.transformer.state_dict(),
        "sched_state_dict": scheduler.state_dict(),
    }
    if name is None:
        T.save(to_save, f"{save_loc}/Forecaster-{today}.pth")
    else:
        T.save(to_save, f"{save_loc}/Forecaster-{name}.pth")


def load_model_parameters(weight_loc: str) -> dict:
    """
    Load training weights, etc.
    """
    checkpoint_dict = T.load(weight_loc)
    return checkpoint_dict


def get_grad_l2_norm(model):
    total_norm: float = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5
