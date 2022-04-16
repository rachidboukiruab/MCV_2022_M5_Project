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
        # TODO
        img_features = self.img_features[index]  # (31014,)
        text_features = self.text_features[index]  # (5, W, 300)

        # maybe useful for the future

        return (image, caption, negative_caption), (caption, image, negative_image)

    def __len__(self):
        return self.img_features.shape[1]
