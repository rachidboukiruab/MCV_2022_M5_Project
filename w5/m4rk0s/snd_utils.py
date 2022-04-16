import numpy as np
import scipy.io
import torch.utils.data as torchdata


class MyDataset(torchdata.Dataset):
    def __init__(self, img_features_file: str, text_features_file: str):
        assert img_features_file.split('/')[-1].split('.')[-1] == 'mat' and \
               text_features_file.split('/')[-1].split('.')[-1] == 'npy', 'img`s features must be .mat & text .npy'
        self.img_features = scipy.io.loadmat(img_features_file)
        self.text_features = np.load(text_features_file)

    def __getitem__(self, index):
        img_features = self.img_features[index]  # (31014,)
        text_features = self.text_features[index]  # (5, 1, 300)

        # maybe useful for the future

        return img_features, text_features, label

    def __len__(self):
        return self.img_features.shape[1]
