import torch
from torch.utils.data import DataLoader
from models import FasterRCNN
from dataset import ImageData
import numpy as np
import sys
import json
import os
from torch.nn import Module, Linear, ReLU, init, Sequential
from torchvision import models


data_path = '/home/aharris/shared/m5/Flickr30k/dataset'

output_path = "/home/aharris/shared/m5/w5/results/task_c"


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    data = ImageData(data_path)
    dataloader = DataLoader(data)


    faster = FasterRCNN()
    faster.to(device)
    print(faster)

    print('--------Extracting visual features---------------')

    feats = np.empty((len(os.listdir(data_path)),4096))
    with torch.no_grad():
      faster.eval()
      for idx, imgs in enumerate(dataloader):
          print(idx)
          imgs = imgs.to(device)
          feats[:,idx] = faster(imgs).cpu().numpy()

    with open('{}/imgfeatures_resnet.npy'.format(output_path), "wb") as f:
        np.save(f, feats)       
    print('Image features from dataset saved.')
