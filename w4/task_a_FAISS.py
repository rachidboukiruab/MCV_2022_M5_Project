from pathlib import Path

import faiss
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import create_headless_resnet18


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def build_index(model, train_dataset, d=32):
    index = faiss.IndexFlatL2(d)  # build the index

    xb = np.empty((len(train_dataset), d))
    find_in_train = dict()
    for ii, (data, label) in enumerate(train_dataset):
        find_in_train[ii] = (data, label)
        xb[ii, :] = model(data.unsqueeze(0)).squeeze().detach().numpy()

    xb = np.float32(xb)
    index.add(xb)  # add vectors to the index

    return index, find_in_train


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path("/home/group01/mcv/datasets/MIT_split")
    EMBED_SHAPE = 32

    transfs_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    train_data = ImageFolder(str(data_path / "train"), transform=transfs_t)
    test_data = ImageFolder(str(data_path / "test"), transform=transfs_t)

    model = create_headless_resnet18(EMBED_SHAPE)
    model = model[:9]
    index, find_in_train = build_index(model, test_data, d=512)

    k = 11  # we want to see 10 nearest neighbors
    query_data = np.empty((len(test_data), 512))

    pred_labels_list = list()
    gt_label_list = list()
    metrics_list = list()
    with torch.no_grad():
        for ii, (img, label) in enumerate(test_data):
            xq = model(img.unsqueeze(0)).squeeze().numpy()
            xq = np.float32(xq)
            metrics, pred_label = index.search(np.array([xq]), k)
            pred_labels_list.append(pred_label)
            gt_label_list.append(label)
            metrics_list.append(metrics)

    PLOT = False
    if PLOT:
        plot_samples = 3
        fig, axs = plt.subplots(plot_samples, k)

        print(f"first {plot_samples}-th samples: ", pred_labels_list[:plot_samples])
        for row in range(plot_samples):
            axs[row, 0].imshow(test_data[row][0].permute((1, 2, 0)).numpy())  # plots query img
            for column in range(1, k):
                img_aux = find_in_train[pred_labels_list[row][0][column]][0].permute((1, 2, 0))
                axs[row, column].imshow(img_aux.numpy())
                print(f"for img {row}, nn id: {pred_labels_list[row][0][column]}")

        plt.title(f'{k} nearest imgs for firts {plot_samples}-th images (FAISS)')
        plt.savefig("./results/jupytest/faiss.png")

    # EVAL

    # print(pred_labels_list[0])  # [[  0 202 320 542  64]]
    # print(gt_label_list[0])  # 0

    pd_single = list()

    for jj, (pd_labels, gt_labs) in enumerate(zip(pred_labels_list, gt_label_list)):
        id_nn = pd_labels[0][1:5]  # 1st nn

        aux = list()
        for ll in id_nn:
            aux.append(find_in_train[ll][1])
        pd_single.append(aux)

    gt_label_list_copy = list()
    for zz in gt_label_list:
        gt_label_list_copy.append([zz])
    gt_label_list = gt_label_list_copy

    scores_k1 = mapk(gt_label_list, pd_single, k=1)
    scores_k5 = mapk(gt_label_list, pd_single, k=5)
    print("MAP@1: ", scores_k1)
    print("MAP@5: ", scores_k5)
