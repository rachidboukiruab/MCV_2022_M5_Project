import random

import numpy as np
import scipy.io
import torch.utils.data as torchdata


def decay_learning_rate(init_lr, optimizer, epoch):
    """
    decay learning late every 4 epoch
    """
    lr = init_lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Img2TextDataset(torchdata.Dataset):
    def __init__(self, img_features_file: str, text_features_file: str):
        assert img_features_file.split('/')[-1].split('.')[-1] == 'mat' and \
               text_features_file.split('/')[-1].split('.')[-1] == 'npy', 'img`s features must be .mat & text .npy'
        self.img_features = scipy.io.loadmat(img_features_file)
        self.text_features = np.load(text_features_file, allow_pickle=True)

    def __getitem__(self, index):
        image = self.img_features[index]  # (31014,)
        # pos_caption = self.text_features[index]  # (5, W, 300)
        positive_cap_sub_id = random.randint(0, self.text_features.shape[1] - 1)
        pos_caption = self.text_features[index][positive_cap_sub_id]

        while True:
            negative_img_id = random.randint(0, self.img_features.shape[1] - 1)
            if negative_img_id != index:
                break

        while True:
            negative_cap_id = random.randint(0, self.text_features.shape[1] - 1)
            if negative_cap_id != index:
                break

        # neg caption extraction
        negative_cap_sub_id = random.randint(0, self.text_features.shape[1] - 1)
        negative_caption = self.text_features[negative_cap_id][negative_cap_sub_id]

        # neg img extraction
        negative_image = self.img_features[negative_img_id]

        return (image, pos_caption, negative_caption), (pos_caption, image, negative_image)

    def __len__(self):
        return self.img_features.shape[1]
