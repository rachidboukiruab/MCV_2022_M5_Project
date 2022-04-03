import os
from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from models import create_headless_resnet18
from utils import print_colored, COLOR_WARNING
from sklearn.metrics import average_precision_score

def build_net(device, d=64):
    model = torchvision.models.resnet50(pretrained=True, progress=True)
    model.eval()
    model.fc = nn.Linear(in_features=2048, out_features=d)
    model = model.to(device)

    return model


def build_index(model, train_dataset, d=32):
    index = faiss.IndexFlatL2(d)  # build the index

    xb = np.empty((len(train_dataset), d))
    find_in_train = dict()
    for ii, (data, label) in enumerate(train_dataset):
        xb[ii, :] = model(data.unsqueeze(0)).squeeze().detach().numpy()
        find_in_train[ii] = (data, label)

    xb = np.float32(xb)
    index.add(xb)  # add vectors to the index

    return index, find_in_train


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
    model = model[:9]
    index, find_in_train = build_index(model, test_data, d=512)

    k = 5  # we want to see 5 nearest neighbors
    query_data = np.empty((len(test_data), 512))

    pred_labels_list = list()
    gt_label_list = list()
    metrics_list = list()
    with torch.no_grad():
        for ii, (img, label) in enumerate(test_data):
            xq = model(img.unsqueeze(0)).squeeze().numpy()
            xq = np.float32(xq)
            metrics, pred_label = index.search(np.array([xq]), k)
            pred_labels_list.append(pred_label)
            gt_label_list.append(label)
            metrics_list.append(metrics)

    PLOT = False
    if PLOT:
        plot_samples = 3
        fig, axs = plt.subplots(plot_samples, k)

        print(f"first {plot_samples}-th samples: ", pred_labels_list[:plot_samples])
        for row in range(plot_samples):
            axs[row, 0].imshow(test_data[row][0].permute((1, 2, 0)).numpy())  # plots query img
            for column in range(1, k):
                axs[row, column].imshow(find_in_train[pred_labels_list[row][0][column]][0].permute((1, 2, 0)).numpy())
                print(f"for img {row}, nn id: {pred_labels_list[row][0][column]}")

        plt.title(f'{k} nearest imgs for firts {plot_samples}-th images (FAISS)')
        plt.savefig("./results/jupytest/faiss.png")

    # EVAL

    print(pred_labels_list[0])
    print(gt_label_list[0])

    # for jj, (pd_labels, gt_labs) in enumerate(zip(pred_labels_list, gt_label_list)):




