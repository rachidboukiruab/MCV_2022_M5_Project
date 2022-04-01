import pickle

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import os

def create_headless_resnet18():
    model = models.resnet18(pretrained=True, progress=False)
    model = nn.Sequential(*list(model.children())[:-1])
    return model

model = create_headless_resnet18()

data_path = Path("/home/group01/mcv/datasets/MIT_split")
feature_path = Path("./results/retrieval")
os.makedirs(feature_path, exist_ok=True)

if __name__ == '__main__':
    transfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    catalogue = ImageFolder(str(data_path / "train"), transform=transfs)
    queries = ImageFolder(str(data_path / "test"), transform=transfs)

    catalogue_meta = [(x[0].split('/')[-1], x[1]) for x in catalogue.imgs]
    query_meta = [(x[0].split('/')[-1], x[1]) for x in queries.imgs]

    with (feature_path / "catalogue_meta.pkl").open('wb') as f_meta:
        pickle.dump(catalogue_meta, f_meta)

    with (feature_path / "query_meta.pkl").open('wb') as f_meta:
        pickle.dump(query_meta, f_meta)


    catalogue_data = np.empty((len(catalogue), 512))
    with torch.no_grad():
        for ii, (img, _) in enumerate(catalogue):
            catalogue_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()

    with open(feature_path / "catalogue.npy", "wb") as f:
        np.save(f, catalogue_data)

    query_data = np.empty((len(queries), 512))
    with torch.no_grad():
        for ii, (img, _) in enumerate(queries):
            query_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()

    with open(feature_path / "queries.npy", "wb") as f:
        np.save(f, query_data)