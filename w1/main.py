import math
import sys

import wandb
import torch
import json

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch import nn, optim
from torch import Tensor
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import *
from typing import TypedDict, Dict, Optional, Any, List

import models


class ExperimentSettings(TypedDict):
    """
	A typed dict to represent experiment settings. Types should match those in
	the configuration JSON file used as input parameter.
	"""
    exp_name: str
    data_path: Path
    out_path: Path
    load_weights: Optional[Path]
    freeze: bool
    batch_size: int
    lr: float
    epochs: int
    model: str
    model_params: Dict[str, Any]
    classes: int
    wandb_project: str
    wandb_entity: str


def setup() -> ExperimentSettings:
    """
	Creates a parser to load a configuration file and returns it as an
	ExperimentSettings dictionary.

	Returns
	-------
	ExperimentSettings
		Dictionary with all experiment-related variables
	"""
    parser = ArgumentParser(
        description='Torch-based image classification system',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the configuration file for the experiment",
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as f_params:
        exp: ExperimentSettings = json.load(f_params)

    # Convert strings to path type for commodity
    exp["data_path"] = Path(exp["data_path"])
    exp["out_path"] = Path(exp["out_path"])
    exp["load_weights"] = Path(exp["load_weights"]) \
        if exp["load_weights"] is not None else None

    return exp


def main(exp: ExperimentSettings) -> None:
    wandb.init(project=exp["wandb_project"], entity=exp["wandb_entity"])
    wandb.config = exp
    train_data = ImageFolder(str(exp["data_path"] / "train"))
    test_data = ImageFolder(str(exp["data_path"] / "test"))

    if str(exp["model"]) == "simplenet":
        model = models.SimpleNet(int(exp["classes"]))
    else:
        raise SystemExit('model name not found')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_model(exp, train_data, model, device)


def train_model(exp, train_data, model, device):
    model = model.to(device)

    # TODO choose between SGD & Adam
    optimizer = optim.SGD(model.parameters(), lr=int(exp["lr"]), momentum=int(exp["momentum"]))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(int(exp["epochs"])):

        # wandb.log({"loss": loss})
        for i, tdata in enumerate(train_data):

            # get imgs & labels -> to GPU/CPU
            data, labels = tdata
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)

            # w&b logger
            if i % wandb.config.log_interval == 0:
                wandb.log({"loss": loss})

            # stop if cracks (?)
            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))
                sys.exit(1)

            loss.backward()
            lr_scheduler.step()


@torch.no_grad()
def test_model():
    pass


if __name__ == "__main__":
    main(setup())
