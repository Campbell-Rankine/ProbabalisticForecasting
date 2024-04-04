import torch as T
from datetime import datetime


from ProbabalisticForecaster import ProbForecaster

save_loc = "./weights"  # change if need be

today = datetime.now().strftime("%Y-%m-%d")


def save_model_params(
    model: ProbForecaster, optimizer: T.optim.AdamW, losses: list, epoch: int
):
    """
    Save full training run including losses, optimizer etc.
    """
    # create save dict
    to_save = {
        "epoch": epoch,
        "loss": losses,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_state_dict": model.state_dict(),
    }
    T.save(to_save, f"{save_loc}/Forecaster-{today}.pth")
