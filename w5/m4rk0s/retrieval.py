import os.path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader
from utils import mpk, mAP
import json
from sklearn.neighbors import KNeighborsClassifier
import pickle
import faiss
from dataset import Img2TextDataset, Text2ImgDataset
from models import ImgEncoder, TextEncoder

img_features_file = '/home/group01/mcv/datasets/Flickr30k/vgg_feats.mat'
text_features_file = '/home/group01/mcv/datasets/Flickr30k/fasttext_feats.npy'


def main(config):
    # TODO: split .npy & .mat into train/test/val
    # TODO: dataset correct
    # TODO: faiss store labels for mpk

    data_path = config['data_path']
    out_path = config['data_path']
    type_of_retrieval = config['type']

    image_model = ImgEncoder()
    text_model = TextEncoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # val dataset
    if type_of_retrieval == 'task_a':
        val_set = Img2TextDataset(img_features_file, text_features_file)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
        checkpoint = torch.load(os.path.join(out_path, 'task_a/models/Image2Text_weights.pth'))
    else:
        val_set = Text2ImgDataset(img_features_file, text_features_file)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
        checkpoint = torch.load(os.path.join(out_path, 'task_b/models/Text2Image_weights.pth'))

    # load model's weights
    image_model.load_state_dict(checkpoint[0])
    text_model.load_state_dict(checkpoint[1])
    image_model.to(device)
    text_model.to(device)
    image_model.eval()
    text_model.eval()

    # create indx
    index = faiss.IndexFlatL2(1000)
    with torch.no_grad():
        if type_of_retrieval == 'task_a':
            for ii, (img, caption) in enumerate(val_dataloader):
                xb = image_model(img).squeeze().numpy()
                xb = np.float32(xb)
                index.add(xb)
        else:
            for ii, (img, caption) in enumerate(val_dataloader):
                xb = text_model(caption).squeeze().numpy()
                xb = np.float32(xb)
                index.add(xb)

    # FAISS retrieval
    k = 5
    pred_label_all = []
    metrics_all = []
    with torch.no_grad():
        if type_of_retrieval == 'task_a':
            for ii, (img, caption) in enumerate(val_dataloader):
                xq = image_model(caption).squeeze().numpy()
                xq = np.float32(xq)
                metrics, pred_label = index.search(np.array([xq]), k)
                pred_label_all.append(pred_label)
                metrics_all.append(metrics)
        else:
            for ii, (img, caption) in enumerate(val_dataloader):
                xq = text_model(caption).squeeze().numpy()
                xq = np.float32(xq)
                metrics, pred_label = index.search(np.array([xq]), k)
                pred_label_all.append(pred_label)
                metrics_all.append(metrics)

    p_1 = mpk(gt_label_list, pd_single, 1)
    p_5 = mpk(gt_label_list, pd_single, 5)
    print('P@1={:.3f}'.format(p_1 * 100))
    print('P@5={:.3f}'.format(p_5 * 100))

    map = mAP(gt_label_list, pd_single)
    print('mAP={:.3f}'.format(map * 100))
    time_list = np.asarray(time_list)
    print(f"FAISS mean TIME{np.mean(time_list)}")


if __name__ == "__main__":
    config = {
        "data_path": "/home/aharris/shared/m5/Flickr30k",
        "out_path": "./results/",
        "type": "task_a"
    }
    main(config)
