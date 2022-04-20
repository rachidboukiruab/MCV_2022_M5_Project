from torch.utils.data import Dataset
import json
import random
import numpy as np
from scipy.io import loadmat

from utils import reduce_txt_embeds
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms


class Img2TextDataset(Dataset):
    def __init__(self, img_features_file: str, text_features_file: str, mode: str):
        assert img_features_file.split('/')[-1].split('.')[-1] == 'mat' and \
               text_features_file.split('/')[-1].split('.')[-1] == 'npy', 'img`s features must be .mat & text .npy'
        assert mode == 'train' or mode == 'test' or mode == 'val'
        self.img_features = loadmat(img_features_file)['feats']
        self.text_features = np.load(text_features_file, allow_pickle=True)
        self.text_features = reduce_txt_embeds(self.text_features)

        # split depending on mode
        if mode == 'train':
            self.img_features = self.img_features[:, :29000]
            self.text_features = self.text_features[:29000, :]
        elif mode == 'val':
            self.img_features = self.img_features[:, 29000:30014]
            self.text_features = self.text_features[2900:30014, :]
        else:
            self.img_features = self.img_features[:, 30014:-1]
            self.text_features = self.text_features[30014:-1, :]

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
    def __init__(self, img_features_file: str, text_features_file: str, mode: str):
        assert img_features_file.split('/')[-1].split('.')[-1] == 'mat' and \
               text_features_file.split('/')[-1].split('.')[-1] == 'npy', 'img`s features must be .mat & text .npy'
        assert mode == 'train' or mode == 'test' or mode == 'val'
        self.img_features = loadmat(img_features_file)['feats']
        self.text_features = np.load(text_features_file, allow_pickle=True)
        self.text_features = reduce_txt_embeds(self.text_features)

        # split depending on mode
        if mode == 'train':
            self.img_features = self.img_features[:, :29000]
            self.text_features = self.text_features[:29000, :]
        elif mode == 'val':
            self.img_features = self.img_features[:, 29000:30014]
            self.text_features = self.text_features[29000:30014, :]
        else:
            self.img_features = self.img_features[:, 30014:-1]
            self.text_features = self.text_features[30014:-1, :]

    def __getitem__(self, index):
        image = self.img_features[:, index]  # (4096,)
        # pos_caption = self.text_features[index]  # (5, W, 300)
        positive_cap_sub_id = random.randint(0, self.text_features.shape[1] - 1)
        pos_caption = self.text_features[index][positive_cap_sub_id]
        # aux_4_val = self.text_features[index] # returns all captions for that img, useful 4 retrieval eval

        while True:
            negative_img_id = random.randint(0, self.img_features.shape[1] - 1)
            if negative_img_id != index:
                break

        # neg img extraction
        negative_image = self.img_features[:, negative_img_id]

        return pos_caption, image, negative_image

    def __len__(self):
        return self.img_features.shape[1]


class FlickrImagesAndCaptions(Dataset):
    SPLITS = ["train", "val", "test"]

    def __init__(
            self,
            dataset_path: str,
            split: str
    ):
        root_path = Path(dataset_path)

        assert (root_path / "fasttext_feats.npy").exists(), "No textual features in data dir"
        assert (root_path / "vgg_feats.mat").exists(), "No image features in data dir"
        assert split in self.SPLITS, "Invalid dataset split"
        assert (root_path / f"{split}.json").exists(), "No split data in data dir"

        # To determine partition files, use imgid from partition jsons
        with open(root_path / f"{split}.json", 'r') as f_json:
            split = json.load(f_json)
            indices = self._get_split_indices(split)

        self.img_features = loadmat(str(root_path / "vgg_feats.mat"))['feats'].T[indices]
        self.text_features = np.load(str(root_path / "fasttext_feats.npy"), allow_pickle=True)
        self.text_features = self._mean_reduction(self.text_features)[indices]

    def __getitem__(self, index):
        img_features = self.img_features[index]  # (Images, FeatureSize)
        txt_features = self.text_features[index]  # (Images, FeatureSize)

        return img_features, txt_features

    def __len__(self):
        return self.img_features.shape[0]

    @staticmethod
    def _get_split_indices(json_obj):
        img_ids = []
        for ii in json_obj:
            img_ids.append(ii["imgid"])
        return np.asarray(img_ids)

    @staticmethod
    def _mean_reduction(embeds):
        aux1 = []
        for i in range(len(embeds)):
            aux2 = []
            for sent in embeds[i]:
                aux2.append(np.mean(sent, axis=0))
            aux1.append(aux2)
        return np.asarray(aux1)


class TripletFaster(Dataset):
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


class ImageData(Dataset):
    def __init__(self, directory):
        self.img_path = directory
        self.images = os.listdir(self.img_path)
        self.image_features = np.empty((1024, len(self.images)))
        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_path, self.images[idx]))
        tensor_img = self.tfms(image)
        return tensor_img
