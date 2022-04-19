from torch.utils.data import Dataset
import random
import numpy as np
from scipy.io import loadmat

from utils import reduce_txt_embeds

class Img2TextDataset(Dataset):
    def __init__(self, img_features_file: str, text_features_file: str):
        assert img_features_file.split('/')[-1].split('.')[-1] == 'mat' and \
               text_features_file.split('/')[-1].split('.')[-1] == 'npy', 'img`s features must be .mat & text .npy'
        self.img_features = loadmat(img_features_file)['feats']
        self.text_features = np.load(text_features_file, allow_pickle=True)
        self.text_features = reduce_txt_embeds(self.text_features)

    def __getitem__(self, index):
        image = self.img_features[:, index]  # (4096,)
        # pos_caption = self.text_features[index]  # (5, W, 300)
        positive_cap_sub_id = random.randint(0, self.text_features.shape[1] - 1)
        pos_caption = self.text_features[index][positive_cap_sub_id]

        while True:
            negative_cap_id = random.randint(0, self.text_features.shape[1] - 1)
            if negative_cap_id != index:
                break

        # neg caption extraction
        negative_cap_sub_id = random.randint(0, self.text_features.shape[1] - 1)
        negative_caption = self.text_features[negative_cap_id][negative_cap_sub_id]

        return image, pos_caption, negative_caption

    def __len__(self):
        return self.img_features.shape[1]


class Text2ImgDataset(Dataset):
    def __init__(self, img_features_file: str, text_features_file: str):
        assert img_features_file.split('/')[-1].split('.')[-1] == 'mat' and \
               text_features_file.split('/')[-1].split('.')[-1] == 'npy', 'img`s features must be .mat & text .npy'
        self.img_features = loadmat(img_features_file)['feats']
        self.text_features = np.load(text_features_file, allow_pickle=True)
        self.text_features = reduce_txt_embeds(self.text_features)

    def __getitem__(self, index):
        image = self.img_features[:, index]  # (4096,)
        # pos_caption = self.text_features[index]  # (5, W, 300)
        positive_cap_sub_id = random.randint(0, self.text_features.shape[1] - 1)
        pos_caption = self.text_features[index][positive_cap_sub_id]

        while True:
            negative_img_id = random.randint(0, self.img_features.shape[1] - 1)
            if negative_img_id != index:
                break

        # neg img extraction
        negative_image = self.img_features[:, negative_img_id]

        return pos_caption, image, negative_image

    def __len__(self):
        return self.img_features.shape[1]


class FasterDataset(Dataset):
    def __init__(self, img_features, text_features_file: str):
        assert text_features_file.split('/')[-1].split('.')[-1] == 'npy', 'img`s features must be .mat & text .npy'
        self.text_features = np.load(text_features_file, allow_pickle=True)
        self.text_features = reduce_txt_embeds(self.text_features)
        self.img_features = img_features

    def __getitem__(self, index):
        image = self.img_features[:, index]  # (4096,)
        # pos_caption = self.text_features[index]  # (5, W, 300)
        positive_cap_sub_id = random.randint(0, self.text_features.shape[1] - 1)
        pos_caption = self.text_features[index][positive_cap_sub_id]

        while True:
            negative_img_id = random.randint(0, self.img_features.shape[1] - 1)
            if negative_img_id != index:
                break

        # neg img extraction
        negative_image = self.img_features[:, negative_img_id]

        return pos_caption, image, negative_image

    def __len__(self):
        return self.img_features.shape[1]
