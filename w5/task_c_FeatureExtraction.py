import torch
from torch.utils.data import DataLoader
from models import FasterRCNN
from dataset import ImageData
import numpy as np
import sys
from PIL import Image

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook



text_features_file = '/home/aharris/shared/m5/Flickr30k/fasttext_feats.npy'
train_path = '/home/aharris/shared/m5/Flickr30k/train/'
test_path = '/home/aharris/shared/m5/Flickr30k/test/'
val_path = '/home/aharris/shared/m5/Flickr30k/val/'
output_path = "./results/task_a/"


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = ImageData(train_path)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)

    test_data = ImageData(test_path)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    val_data = ImageData(val_path)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

    faster = FasterRCNN()
    faster.to(device)
    print(faster)
    
    faster.fc.register_forward_hook(get_features('features'))
    print('--------Extracting visual features for the training set---------------')
    train_feats, test_feats, val_feats = [],[],[]

    features = {}
    for idx, imgs in enumerate(train_dataloader):
        imgs = imgs.to(device)
        out = faster(imgs)
        train_feats.append(features['features'].cpu().numpy())
        
    with open('{}/train_imgfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, train_feats)       
    print('Image features from Train dataset saved.')

    faster.fc.register_forward_hook(get_features('features'))
    print('--------Extracting visual features for the validation set---------------')
    features = {}
    for idx, imgs in enumerate(val_dataloader):
        imgs = imgs.to(device)
        out = faster(imgs)
        val_feats.append(features['features'].cpu().numpy())

    with open('{}/val_imgfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, val_feats)       
    print('Text features from Validation dataset saved.')


    faster.fc.register_forward_hook(get_features('features'))
    print('--------Extracting visual features for the test set---------------')
    features = {}
    for idx, imgs in enumerate(test_dataloader):
        imgs = imgs.to(device)
        out = faster(imgs)
        test_feats.append(features['features'].cpu().numpy())

    with open('{}/test_imgfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, test_feats)       
    print('Text features from test dataset saved.')

