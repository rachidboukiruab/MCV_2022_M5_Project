import torch
import torchvision
from torch import nn
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_net(device, d = 64):
    model = torchvision.models.resnet50(pretrained=True, progress=True)
    model.eval()
    model.fc = nn.Linear(in_features=2048, out_features=d)
    model = model.to(device)

    return model


def build_index(model, train_dataset, d=64):
    # USES GPU!!

    res = faiss.StandardGpuResources()  # defines resource, use a single GPU
    index = faiss.IndexFlatL2(d)  # build the index
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index) # make it into a gpu index
    print(gpu_index.is_trained)

    for data in train_dataset:
        xb = model(data)
        gpu_index.add(xb)  # add vectors to the index

        # SANITY TODO: remove this after debugging
        D, I = gpu_index.search(xb[:5], 4)  # sanity check
        print(
            "As a sanity check, we can first search a few database vectors, to make sure the nearest neighbor is indeed the vector itself.")
        print(I)
        print(D)

    return gpu_index


def search_faiss(index, query, k=4):
    D, I = index.search(query, k)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries


if __name__ == '__main__':
    # TODO load dataset
