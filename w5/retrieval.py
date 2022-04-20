import os.path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Img2TextDataset, Text2ImgDataset
from models import ImgEncoder, TextEncoder
from utils import mAP


def main(config):
    # TODO: faiss store labels for mpk

    data_path = config['data_path']
    out_path = config['out_path']
    type_of_retrieval = config['type']

    img_features_file = os.path.join(data_path, 'vgg_feats.mat')
    text_features_file = os.path.join(data_path, 'fasttext_feats.npy')

    image_model = ImgEncoder()
    text_model = TextEncoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # val dataset
    if type_of_retrieval == 'task_a':
        val_set = Img2TextDataset(img_features_file, text_features_file, mode='val')
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
        checkpoint = torch.load(os.path.join(out_path, 'task_a/models/Image2Text_weights.pth'))
    else:
        val_set = Text2ImgDataset(img_features_file, text_features_file, mode='val')
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
    d = 1000
    index = faiss.IndexFlatL2(d)
    xb = np.empty((len(val_set), d))
    with torch.no_grad():
        if type_of_retrieval == 'task_a':
            for ii, (img, caption, _) in enumerate(val_dataloader):
                img.to(device)
                print(img)
                print(image_model)
                xb[ii, :] = image_model(img).squeeze().numpy()

        else:
            for ii, (caption, img, _) in enumerate(val_dataloader):
                caption.to(device)
                xb[ii, :] = text_model(caption).squeeze().numpy()

    xb = np.float32(xb)
    index.add(xb)

    # FAISS retrieval
    k = 5
    pred_label_all = []
    with torch.no_grad():
        if type_of_retrieval == 'task_a':
            for ii, (img, pos_caption, _) in enumerate(val_dataloader):
                caption.to(device)
                xq = text_model(caption).squeeze().numpy()
                xq = np.float32(xq)
                _, pred_label = index.search(np.array([xq]), k)
                print(pred_label)
                pred = 0
                for lab in pred_label:
                    if lab == img:
                        pred = 1
                pred_label_all.append(pred)
        else:
            for ii, (caption, img, _) in enumerate(val_dataloader):
                img.to(device)
                xq = image_model(img).squeeze().numpy()
                xq = np.float32(xq)
                _, pred_label = index.search(np.array([xq]), k)
                print(pred_label)
                pred = 0
                for lab in pred_label:
                    for cap in caption:
                        if lab == cap:
                            pred = 1
                pred_label_all.append(pred)

    ground_truth = np.ones_like(pred_label_all)

    # p_1 = mpk(ground_truth, pred_label_all, 1)
    # p_5 = mpk(ground_truth, pred_label_all, 5)
    # print('P@1={:.3f}'.format(p_1 * 100))
    # print('P@5={:.3f}'.format(p_5 * 100))

    map = mAP(ground_truth, pred_label_all)
    print('mAP={:.3f}'.format(map * 100))


if __name__ == "__main__":
    config = {
        "data_path": "/home/group01/mcv/datasets/Flickr30k",
        "out_path": "./results/",
        "type": "task_a"
    }
    main(config)
