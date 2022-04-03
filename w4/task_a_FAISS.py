from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pickle
from models import create_headless_resnet18
from utils import mpk, mAP


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

    labels_names = ['Open Country', 'Coast', 'Forest', 'Highway', 'Inside City', 'Mountain', 'Street', 'Tall Building']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path("/home/group01/mcv/datasets/MIT_split/")
    EMBED_SHAPE = 32

    transfs_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    train_data = ImageFolder("/home/group01/mcv/datasets/MIT_split/train", transform=transfs_t)
    test_data = ImageFolder("/home/group01/mcv/datasets/MIT_split/test", transform=transfs_t)

    model = create_headless_resnet18(EMBED_SHAPE)
    model = model[:9]
    index, find_in_train = build_index(model, train_data, d=512)

    k = 5  # we want to see 10 nearest neighbors + the img itself
    query_data = np.empty((len(test_data), 512))

    pred_labels_list = list()
    gt_label_list = list()
    metrics_list = list()
    with torch.no_grad():
        for ii, (img, label) in enumerate(test_data):
            if label == 1:
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

    SLIDES = False

    if SLIDES:

        for xz in range(len(pred_labels_list)):
            labels_list_auxz = pred_labels_list[xz][0]
            for xy in range(len(labels_list_auxz)):
                auxxy = labels_list_auxz[xy]
                print(f"query_{xz}_k{xy}: {labels_names[find_in_train[auxxy][1]]}")
                plt.imsave(f"./results/jupytest/slides/query_{xz}_k{xy}.png",
                           find_in_train[auxxy][0].permute((1, 2, 0)).numpy())

    # EVAL

    # print(pred_labels_list[0])  # [[  0 202 320 542  64]]
    # print(gt_label_list[0])  # 0

    pd_single = list()

    for jj, (pd_labels, gt_labs) in enumerate(zip(pred_labels_list, gt_label_list)):
        id_nn = pd_labels[0]  # 1st nn
        aux = list()
        for ll in id_nn:
            aux.append(find_in_train[ll][1])
        pd_single.append(aux)

    # gt_label_list_copy = list()
    # for zz in gt_label_list:
    #     gt_label_list_copy.append([zz])
    # gt_label_list = gt_label_list_copy

    p_1 = mpk(gt_label_list, pd_single, 1)
    p_5 = mpk(gt_label_list, pd_single, 5)
    print('P@1=', p_1)
    print('P@5=', p_5)

    # gt_label_list_copy = list()
    # for zz in gt_label_list:
    #     gt_label_list_copy.append([zz] * k)
    # gt_label_list = gt_label_list_copy

    print(gt_label_list)
    print(pd_single)

    map = mAP(gt_label_list, pd_single)
    print('mAP=', map)

    # # For each class
    # # prepare data
    # pred_labels_list = list()
    # gt_label_list = list()
    # for jj, (pd_labels, gt_labs) in enumerate(zip(pred_labels_list, gt_label_list)):
    #
    # precision = dict()
    # recall = dict()
    # average_precision = dict()
    # for i in range(8):
    #     precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
    #     average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
