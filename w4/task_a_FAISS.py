import os
from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from models import create_headless_resnet18
from utils import print_colored, COLOR_WARNING


def build_net(device, d=64):
    model = torchvision.models.resnet50(pretrained=True, progress=True)
    model.eval()
    model.fc = nn.Linear(in_features=2048, out_features=d)
    model = model.to(device)

    return model


def build_index(model, train_dataset, d=32):
    index = faiss.IndexFlatL2(d)  # build the index

    xb = np.empty((len(train_dataset), d))
    for ii, (data, label) in enumerate(train_dataset):
        xb[ii, :] = model(data.unsqueeze(0)).squeeze().detach().numpy()

    xb = np.float32(xb)
    print(xb.shape)
    index.add(xb)  # add vectors to the index

    # SANITY TODO: remove this after debugging
    D, I = index.search(xb[:5], 4)  # sanity check
    print(
        "As a sanity check, we can first search a few database vectors, to make sure the nearest neighbor is indeed the vector itself.")
    print(I)
    print(D)

    return index


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path("/home/group01/mcv/datasets/MIT_split")
    EMBED_SHAPE = 32

    transfs_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    train_data = ImageFolder(str(data_path / "train"), transform=transfs_t)
    test_data = ImageFolder(str(data_path / "test"), transform=transfs_t)

    model = create_headless_resnet18(EMBED_SHAPE)
    index = build_index(model, test_data)

    k = 5  # we want to see 5 nearest neighbors
    query_data = np.empty((len(test_data), EMBED_SHAPE))

    pred_labels_list = list()
    gt_label_list = list()
    metrics_list = list()
    with torch.no_grad():
        for ii, (img, label) in enumerate(test_data):
            xq = model(img.unsqueeze(0)).squeeze().numpy()
            xq = np.float32(xq)
            print(xq.shape)
            pred_label, metrics = index.search(xq, k)
            pred_labels_list.append(pred_label)
            gt_label_list.append(label)
            metrics_list.append(metrics)
            print(pred_label)
            print(metrics)
            print('--' * 10)
