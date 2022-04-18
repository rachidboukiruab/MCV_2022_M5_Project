import numpy as np

from pathlib import Path
from scipy.io import loadmat
from torch.utils import data


class FlickrImagesAndCaptions(data.Dataset):
    def __init__(
            self,
            data_path: Path,
            split: str,
            sentence_embed: str
    ):
        self.img_features = loadmat(str(data_path / 'vgg_feats.mat'))['feats']  # (31014, 5, W, 300)
        self.txt_features = np.load(
            str(data_path / 'fasttext_feats.npy'),
            allow_pickle=True
        ).T     # (31014, 4096)

        self.txt_shape = self.txt_features.shape
        self.img_shape = self.img_features.shape


    def __getitem__(self, item):
        ...

    def __len__(self):
        ...

    @staticmethod
    def _mean_embedder(sentence):
        ...

    @staticmethod
    def _additive_embedder(sentence):
        ...
