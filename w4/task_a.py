import faiss
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import print_colored, COLOR_WARNING

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

    transfs = transforms.Compose([
        transforms.ColorJitter(brightness=.3, hue=.3),
        transforms.RandomResizedCrop(256, (0.15, 1.0)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    transfs_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    train_data = ImageFolder(str(data_path / "train"), transform=transfs)
    test_data = ImageFolder(str(data_path / "test"), transform=transfs_t)

    train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    print_colored(f"(dataset info) train: {len(train_loader) * epochs} images", COLOR_WARNING)
    print_colored(f"(dataset info) test: {len(test_loader) * epochs} images", COLOR_WARNING)

    print_colored(f"(dataset info) train: {len(train_loader)} images in the folder", COLOR_WARNING)
    print_colored(f"(dataset info) test: {len(test_loader)} images in the folder", COLOR_WARNING)

