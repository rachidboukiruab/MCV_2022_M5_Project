import matplotlib.pyplot as plt
import logging

from pathlib import Path

import pytorch_metric_learning
from pytorch_metric_learning import distances
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from pytorch_metric_learning import samplers
from pytorch_metric_learning import trainers
from pytorch_metric_learning import testers
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from scipy.io import loadmat

import numpy as np
import umap
import torch
import torch.nn as nn
from cycler import cycler
from torch import optim

# from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder


from models import Triplet
from dataset import flickrDataset, TripletFlickrDatasetImgToTxt

def mean_words(text_data):
    img_texts = []
    for i in range(len(text_data)):
        sentences = []
        for sent in text_data[i]:
            sentences.append(np.mean(sent, axis=0))
        img_texts.append(sentences)
    return np.asarray(img_texts)

def main(config):
    data_path = Path(config["data_path"])
    output_path = Path(config["out_path"])

    img_features = loadmat(f'{data_path}/vgg_feats.mat')['feats']
    img_features = np.transpose(img_features)
    txt_features = np.load(f'{data_path}/fasttext_feats.npy', allow_pickle=True)
    txt_features = mean_words(txt_features)

    train_data = flickrDataset(img_features,txt_features)
    triplet_train_data = TripletFlickrDatasetImgToTxt(train_data)

    train_loader = DataLoader(triplet_train_data,
                              batch_size=config["batch_size"],
                              shuffle=True)

    img_dimensions = np.asarray(img_features).shape # (31014, 4096) -> (features, images)
    txt_dimensions = txt_features.shape # (31014, 5, 300) -> (images, sentences, features)
    
    model = Triplet(img_dimensions[0], txt_dimensions[2],config["embed_size"])
    #model.load_state_dict(torch.load('/home/aharris/shared/m5/CONTRASTIVE.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # summary(model)

    optimizer = optim.Adam(model.parameters(), 3E-4)

    model = model.to(device)
    
    # Triplet loss
    loss_funcs = losses.TripletMarginLoss(margin=0.1)
    mining_funcs = miners.BatchHardMiner()

    model_folder = str(output_path / "models")



if __name__ == "__main__":
    config = {
        "data_path": "/home/aharris/shared/m5/Flickr30k",
        "out_path": "./results/jupytest/",
        "feature_path": "./results/retrieval/",
        "embed_size": 32,
        "batch_size": 64,
    }
    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)
    main(config)
