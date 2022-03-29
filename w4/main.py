import json
import math
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os.path import join
from pathlib import Path
from typing import TypedDict, Dict, Optional, Any
import numpy as np

import torch
import wandb
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchinfo import summary
from utils import make_dirs, print_colored, COLOR_WARNING, HardNegativePairSelector
from datasets import BalancedBatchSampler
from models import EmbeddingNet
from losses import OnlineContrastiveLoss, OnlineTripletLoss


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
    early_stopping: int
    model: str
    model_params: Dict[str, Any]
    classes: int
    wandb_project: str
    wandb_entity: str
    save_every: int
    architecture: str
    n_samples: int
    margin : int
    log_interval : int

    


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # w&b logger
    wandb.init(
        project=exp["wandb_project"],
        entity=exp["wandb_entity"],
        config=exp
    )

    # load train & test data

    transfs = transforms.Compose([
        #transforms.ColorJitter(brightness=.3, hue=.3),
        #transforms.RandomResizedCrop(256, (0.15, 1.0)),
        #transforms.RandomRotation(degrees=30),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    transfs_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    train_data = ImageFolder(str(exp["data_path"] / "train"), transform=transfs)
    test_data = ImageFolder(str(exp["data_path"] / "test"), transform=transfs_t)

    #kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    online_train_loader = BalancedBatchSampler(train_data.classes, exp["classes"], exp["n_samples"])
    online_test_loader = BalancedBatchSampler(test_data.classes, exp["classes"], exp["n_samples"])


    print_colored(f"(dataset info) train: {len(online_train_loader)*exp['epochs']} images", COLOR_WARNING)
    print_colored(f"(dataset info) test: {len(online_test_loader)*exp['epochs']} images", COLOR_WARNING)

    print_colored(f"(dataset info) train: {len(online_train_loader)} images in the folder", COLOR_WARNING)
    print_colored(f"(dataset info) test: {len(online_test_loader)} images in the folder", COLOR_WARNING)

    # load model
    model = EmbeddingNet()
    model.to(device=device)

    if exp["architecture"] == "siamese":
        loss = OnlineContrastiveLoss(exp["margin"], HardNegativePairSelector())

    if exp["architecture"] == "triplet":
        loss = OnlineTripletLoss(exp["margin"], ) #falta la función de triplet selector

    optimizer = optim.Adam(model.parameters(), lr=exp["lr"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.96
    )

    #summary(model, input_size=(exp["batch_size"], 3, 256, 256))

    # Setup output path stuff
    weight_path = exp["out_path"] / exp["exp_name"]
    make_dirs(weight_path)

    best_model = -1
    best_acc = 0.0
    for epoch in range(exp["epochs"]):
        train_loss = train_model(online_train_loader, model, device, optimizer, exp["log_interval"], loss, lr_scheduler)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, exp["epochs"], train_loss)

        test_loss = eval(online_test_loader, model, loss,device)
        test_loss /= len(online_test_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, exp["epochs"],test_loss)


        # w&b logger
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": lr_scheduler.get_last_lr()[0],
            "validation_loss": test_loss,
        })

        """ # Model saving
        if test_accuracy > best_acc + 0.005:
            best_model = epoch
            best_acc = test_accuracy
            torch.save(
                model.state_dict(),
                str(weight_path / "weights_final.pth")
            ) """

        if epoch % exp["save_every"] == 0:
            torch.save(
                model.state_dict(),
                str(weight_path / f"weights_{epoch}.pth")
            )

        # Early Stopping
        if (epoch - best_model) > exp["early_stopping"]:
            print_colored(f"Early stopping at epoch {epoch}", COLOR_WARNING)
            break

    # sync file with w&b
    if (weight_path / "weights_final.pth").exists():
        wandb.save(str(weight_path / f"weights_final.pth"))

    """ train_embeddings_ocl, train_labels_ocl = extract_embeddings(online_train_loader, model)
    plot_embeddings(train_embeddings_ocl, train_labels_ocl, f'{exp["architecture"]}_embeddings_train.jpg')
    test_embeddings_ocl, test_labels_ocl = extract_embeddings(online_test_loader, model)
    plot_embeddings(test_embeddings_ocl, test_labels_ocl, f'{exp["architecture"]}_embeddings_test.jpg') """


def train_model(train_loader, model, device, optimizer, log_interval, loss_fn, lr_scheduler):
    """
    Trains 1 epoch
    """

    model.train()
    losses = []
    running_loss = 0.0
    
    for  batch_idx, (data, target) in enumerate(train_loader):

        # get imgs & labels -> to GPU/CPU
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
       
        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)

        optimizer.zero_grad()

        output = model(*data)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        running_loss += loss.item()

        # stop if cracks (?)
        if not math.isfinite(loss):
            print_colored("Loss is {}, stopping training".format(loss), "0;30;43")
            sys.exit(1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))

            print(message)

            losses = []

        running_loss/=(batch_idx + 1)
    lr_scheduler.step()
    return running_loss, lr_scheduler


@torch.no_grad()
def eval(test_loader, model, loss_fn, device):

    running_loss = 0.0
    model = model.to(device)
    model.eval()

    for batch_idx, (data, target) in enumerate(test_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)

            data = tuple(d.to(device) for d in data)
            if target is not None:
                target = target.to(device)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            running_loss += loss.item()

    return running_loss


if __name__ == "__main__":
    main(setup())
