import wandb
import torch
import json

from torch import nn
from torch import Tensor
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import *


wandb.init(project="m5project", entity="m5project")
wandb.config = {
	# Settings here
}

DATASET_PATH = Path("/home/pautorras/Documents/master/data/MIT_split")

train_data = ImageFolder(str(DATASET_PATH / "train"))
test_data  = ImageFolder(str(DATASET_PATH / "test"))

for epoch in range(MAX_EPOCHS):
	loss = 0.0

	for i, data in enumerate(train_data):
		wandb.log({"loss": loss})

