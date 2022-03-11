import json
import math
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os.path import join
from pathlib import Path
from typing import TypedDict, Dict, Optional, Any

import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import models
from utils import make_dirs, colorize_string, print_colored

transfs = transforms.Compose([
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(0, 45)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
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

    # create path
    make_dirs(str(exp["save_path"]))

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

    # print(f"(dataset info) train: {len(train_loader)} images")
    # print(f"(dataset info) test: {len(test_loader)} images")
    print_colored(f"(dataset info) train: {len(train_loader)*exp['epochs']} images", "0;30;43")
    print_colored(f"(dataset info) test: {len(test_loader)*exp['epochs']} images", "0;30;43")

    # load model
    if str(exp["model"]) == "smallnet":
        model = models.SmallNet(exp["classes"])
        print("Using smallnet...")
    else:
        raise SystemExit('model name not found')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(exp["epochs"]):
        # print(f"DB: epoch {epoch}")
        train_loss, train_accuracy, lr_scheduler = train_model(exp, train_loader, model, device)
        test_loss, test_accuracy = eval(test_loader, model, device)
        print(lr_scheduler.get_last_lr())
        # w&b logger
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader.dataset),
            "learning_rate": lr_scheduler.get_last_lr(),
            "validation_loss": test_loss / len(test_loader.dataset),
            "train_accuracy": train_accuracy,
            "validation_accuracy": test_accuracy,
        })

    # creates path to store weights
    PATH = join(str(exp['save_path']), wandb.run.name)
    make_dirs(PATH)
    PATH = join(PATH, 'model_weights.pth')

    # saves weights
    torch.save(model.state_dict(), PATH)

    # sync file with w&b
    wandb.save(PATH)
    # 4 loading:
    # wandb.restore('model_weights.pth', run_path="lavanyashukla/save_and_restore/10pr4joa")
    # model = ()
    # model.load_state_dict(torch.load('model_weights.pth'))


def train_model(exp, train_loader, model, device):
    """
    Trains 1 epoch
    """

    model = model.to(device)
    model.train()

    # TODO choose between SGD & Adam
    # optimizer = optim.SGD(model.parameters(), lr=exp["lr"], momentum=exp["momentum"])
    optimizer = optim.Adam(model.parameters(), lr=exp["lr"], weight_decay=exp["weight_decay"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=1,
                                                   gamma=0.95)
    criterion = torch.nn.CrossEntropyLoss()

    running_loss = 0.0
    correct, total = 0, 0
    for i, tdata in enumerate(train_loader):

        # get imgs & labels -> to GPU/CPU
        data, labels = tdata
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, labels)
        # print(loss)
        # print(f"Labels{labels}")
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

        # stop if cracks (?)
        if not math.isfinite(loss):
            print_colored("Loss is {}, stopping training".format(loss), "0;30;43")
            sys.exit(1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

    return running_loss, correct / total, lr_scheduler


@torch.no_grad()
def eval(test_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    for tsdata in test_loader:
        tsdata, tslabels = tsdata
        tsdata, tslabels = tsdata.to(device), tslabels.to(device)

        outs = model(tsdata)
        test_loss = criterion(outs, tslabels)

        _, predicted = torch.max(outs.data, 1)
        total += tslabels.size(0)
        correct += (predicted == tslabels).sum().item()

        running_loss += test_loss.item()

    return running_loss, correct / total


if __name__ == "__main__":
    main(setup())
