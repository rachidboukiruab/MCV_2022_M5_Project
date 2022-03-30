# TODO colourful prints
import os
#import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import combinations


COLOR_WARNING = "\x1b[0;30;43m"
MIT_split_classes = ['0', '1', '2', '3', '4', '5', '6', '7']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']



def make_dirs(path):
    """
    check if dir exists, if not: creates it
    :param path: path to create
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creating path {path}")


def print_colored(string: str, color_id):
    """
    prints a string colorized
    :param string: string to colorize
    :param color_id: reference https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
    :return:
    """
    print(color_id + string + '\x1b[0m')


def colorize_string(string: str, color_id):
    """
    colorizes a string to use with "print" function
    :param string: string to colorize
    :param color_id: color_id reference: https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
    :return: concatenated string with colorize format
    """
    return color_id + string + '\x1b[0m'

""" def plot_embeddings(embeddings, targets,filename, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(MIT_split_classes)
    plt.savefig(f'./results/{filename}')

def extract_embeddings(device, dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if device:
            images = images.to(device=device)
            embeddings[k:k+len(images)] = model.get_embedding(images).data.to(device=device).numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels """

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError



class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs
