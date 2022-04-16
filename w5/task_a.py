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


from models import TripletLossModel
from dataset import flickrDataset, TripletFlickrDatasetImgToTxt

def decay_learning_rate(init_lr, optimizer, epoch):
    """
    decay learning late every 4 epoch
    """
    lr = init_lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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


    # Triplet loss
    img_dimensions = np.asarray(img_features).shape # (31014, 4096) -> (features, images)
    txt_dimensions = txt_features.shape # (31014, 5, 300) -> (images, sentences, features)
    
    init_lr = 3E-4
    optimizer = optim.Adam(model.parameters(), init_lr)
    loss_func = losses.TripletMarginLoss(margin=0.1)

    model = TripletLossModel(img_dimensions[0], 
                    txt_dimensions[2],
                    config["embed_size"],
                    optimizer,
                    loss_func)
    #model.load_state_dict(torch.load('/home/aharris/shared/m5/CONTRASTIVE.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # summary(model)

    model = model.to(device)

    model_folder = str(output_path / "models")

    model.train_start()
    for epoch in range(12):
        decay_learning_rate(init_lr, model.optimizer, epoch)

        for i_batch, batch in enumerate(train_loader):
            image_triple, caption_triple = batch
            if device == "cuda":
                image_triple = image_triple.cuda()
                caption_triple = caption_triple.cuda()

            loss = model.forward(image_triple, caption_triple)
            print(f'epoch: {epoch}\titeration: {i_batch}\tLoss: {loss}')

    torch.save(model.state_dict(), '{0}/Image2Text.model'.format(model_folder))


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
