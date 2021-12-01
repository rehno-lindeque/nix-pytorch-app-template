from app.train.common import Epoch, DirPath
from dataclasses import dataclass
from typing import Any, Dict
import dataclasses
import glob
import json
import os
import torch

# Any pytorch state_dict
CheckpointDict = Dict[str, Any]

# A dictionary mapping metrics (e.g. "loss") to measurements
# TODO instanceof Measurements
Measurements = Dict[str, Any]


@dataclass(frozen=True)
class Checkpoint:
    """
    Training checkpoint:

    Done properly, the output of a training run should be completely determined
    by the provided config + the initial checkpoint + the code revision.

    The checkpoint should be updated as necessary at the end of every training
    step so that training could theoretically be interupted at any point.

    I.e. training epoch ranges [1,25) and then [25,50) over two runs ought to
    yield the same result as training epoch range [1,50) in a single run.

    This should include:
    * RNG state
    * Model parameters
    * Learning rate scheduler (e.g. epoch)
    """

    model: CheckpointDict
    scheduler: CheckpointDict
    optimizer: CheckpointDict


@dataclass(frozen=True)
class State:
    """
    Includes all serializable state, including the latest checkpoint as well as
    meta data such as measurements.

    The meta data such as measurements aren't required for reproducibility, but
    saving it to structured json format has provide additional benefits. For
    example, measurements can be used for selecting the best model programmatically
    or more generally for hyperparameter optimization.
    """

    checkpoint: Checkpoint
    epoch: Epoch
    measurements: Measurements


def load(root_path: DirPath) -> State:
    load_path = max(glob.glob(os.path.join(root_path, "[0-9]" * 5, "")))
    checkpoint_data = torch.load(os.path.join(load_path, "checkpoint.pt"))
    with open(os.path.join(load_path, "metadata.json"), "r") as file:
        metadata = json.load(file)
    return State(checkpoint=Checkpoint(**checkpoint_data), **metadata)


def save(state: State, root_path: DirPath):
    save_path = os.path.join(root_path, str(state.epoch).zfill(5))
    os.makedirs(save_path, exist_ok=True)

    # Save checkpoint information
    torch.save(
        dataclasses.asdict(state.checkpoint), os.path.join(save_path, "checkpoint.pt")
    )

    # Save remaining metadata separately
    def encoder(x):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 0:
                return x.item()
            else:
                return x.tolist()

    with open(os.path.join(save_path, "metadata.json"), "w") as file:
        json.dump(
            {
                "epoch": state.epoch,
                "measurements": state.measurements,
            },
            file,
            default=encoder,
        )
    pass
