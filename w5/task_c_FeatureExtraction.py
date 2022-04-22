import torch
from torch.utils.data import DataLoader
from models import FasterRCNN
from dataset import ImageData
import numpy as np
import sys
from PIL import Image
import os

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook



text_features_file = '/home/aharris/shared/m5/Flickr30k/fasttext_feats.npy'
train_path = '/home/aharris/shared/m5/Flickr30k/train/'
test_path = '/home/aharris/shared/m5/Flickr30k/test/'
val_path = '/home/aharris/shared/m5/Flickr30k/val/'
output_path = "/home/aharris/shared/m5/w5/results/taskc"



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = ImageData(train_path)
    train_dataloader = DataLoader(train_data)

    test_data = ImageData(test_path)
    test_dataloader = DataLoader(test_data)

    val_data = ImageData(val_path)
    val_dataloader = DataLoader(val_data)

    faster = FasterRCNN()
    faster.to(device)
    print(faster)
    
    faster.fc.register_forward_hook(get_features('features'))
    print('--------Extracting visual features for the training set---------------')
    

    features = {}
    train_feats = np.empty((4096,len(os.listdir(train_path))))
    for idx, imgs in enumerate(train_dataloader):
        print(idx)
        imgs = imgs.to(device)
        out = faster(imgs)
        train_feats[:,idx] = (features['features'].cpu().numpy())

        
        
    with open('{}/train_imgfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, train_feats)       
    print('Image features from Train dataset saved.')

    faster.fc.register_forward_hook(get_features('features'))
    print('--------Extracting visual features for the validation set---------------')
    features = {}
    val_feats = np.empty((4096,len(os.listdir(val_path))))
    for idx, imgs in enumerate(val_dataloader):
        print(idx)
        imgs = imgs.to(device)
        out = faster(imgs)
        val_feats[:,idx]=(features['features'].cpu().numpy())

    with open('{}/val_imgfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, val_feats)       
    print('Text features from Validation dataset saved.')


    faster.fc.register_forward_hook(get_features('features'))
    print('--------Extracting visual features for the test set---------------')
    features = {}
    test_feats = np.empty((4096,len(os.listdir(test_path))))
    for idx, imgs in enumerate(test_dataloader):
        print(idx)
        imgs = imgs.to(device)
        out = faster(imgs)
        test_feats[:,idx]=(features['features'].cpu().numpy())

    with open('{}/test_imgfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, test_feats)       
    print('Text features from test dataset saved.')

