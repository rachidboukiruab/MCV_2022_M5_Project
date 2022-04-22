import torch
from torch.utils.data import DataLoader
from models import FasterRCNN
from dataset import ImageData
import numpy as np
import sys
import json
import os

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook



data_path = '/home/aharris/shared/m5/Flickr30k/dataset'

output_path = "/home/aharris/shared/m5/w5/results/task_c"



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    data = ImageData(data_path)
    dataloader = DataLoader(data)


    faster = FasterRCNN()
    faster.to(device)
    print(faster)
    
    faster.fc.register_forward_hook(get_features('features'))
    print('--------Extracting visual features---------------')
    

    features = {}
    feats = np.empty((4096,len(os.listdir(data_path))))
    for idx, imgs in enumerate(dataloader):
        print(idx)
        imgs = imgs.to(device)
        out = faster(imgs)
        feats[:,idx] = (features['features'].cpu().numpy())

        
        
    with open('{}/imgfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, feats)       
    print('Image features from dataset saved.')
