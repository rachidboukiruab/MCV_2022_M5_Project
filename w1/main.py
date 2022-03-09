import math
import sys

import wandb
import torch
import json

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch import nn, optim
from torch import Tensor
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import *
from typing import TypedDict, Dict, Optional, Any, List

import models

transfs = transforms.Compose([
    transforms.ToTensor()
])


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

    # w&b logger
    wandb.init(
        project=exp["wandb_project"],
        entity=exp["wandb_entity"],
        config=exp
    )

    # load train & test data
    train_data = ImageFolder(str(exp["data_path"] / "train"), transform=transfs)
    test_data = ImageFolder(str(exp["data_path"] / "test"), transform=transfs)

    train_loader = DataLoader(train_data, batch_size=exp["batch_size"], pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=exp["batch_size"], pin_memory=True)

    # load model
    if str(exp["model"]) == "smallnet":
        model = models.SmallNet(exp["classes"])
        print("Using smallnet...")
    else:
        raise SystemExit('model name not found')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_model(exp, train_loader, model, device)


def train_model(exp, train_loader, model, device):
    model = model.to(device)
    model.train()

    # TODO choose between SGD & Adam
    # optimizer = optim.SGD(model.parameters(), lr=exp["lr"], momentum=exp["momentum"])
    optimizer = optim.Adam(model.parameters(), lr=exp["lr"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10000,
                                                   gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(exp["epochs"]):
        # print(f"DB: epoch {epoch}")
        running_loss = 0.0
        for i, tdata in enumerate(train_loader):

            # get imgs & labels -> to GPU/CPU
            data, labels = tdata
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)
            # print(loss)
            # print(f"Labels{labels}")

            running_loss += loss.item()

            # stop if cracks (?)
            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))
                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
        # w&b logger
        wandb.log({
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader.dataset),
            "learning_rate": lr_scheduler.get_last_lr()[0],
        })

@torch.no_grad()
def eval(test_loader, model, device):
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    for tsdata in test_loader:

        tsdata, tslabels = tsdata
        tsdata, tslabels = tsdata.to(device), tslabels.to(device)

        outs = model(tsdata)
        _, predicted = torch.max(outs.data, 1)
        total += tslabels.size(0)
        correct += (predicted == tslabels).sum().item()

    return correct / total


if __name__ == "__main__":
    main(setup())
