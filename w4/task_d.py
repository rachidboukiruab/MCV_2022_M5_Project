from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap.plot
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import create_headless_resnet18

if __name__ == '__main__':

    #########
    min_dist = 0.0001
    n_neighbors = 10

    #########
    data_path = Path("/home/group01/mcv/datasets/MIT_split")
    feature_path = Path("./results/retrieval")
    trained_path = Path("./results/jupytest")

    weights_filename = ["weights.pth", "weights_triplet.pth"]

    model = create_headless_resnet18(32)
    # LOAD PRE_TRAINED WEIGHTS
    model.load_state_dict(torch.load(trained_path / weights_filename[0]))

    transfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    query = ImageFolder(str(data_path / "test"), transform=transfs)
    query_data = np.empty((len(query), 32))

    color_4_umap = list()
    select_color = [(66, 135, 245), (161, 66, 245), (230, 66, 245), (245, 66, 135), (245, 66, 66), (245, 179, 66), (203, 245, 66), (66, 245, 75)]
    with torch.no_grad():
        for ii, (img, label) in enumerate(query):
            query_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()
            print(type(label))
            color_4_umap.append(select_color[label])


    print(f"QUERY SHAPE {query_data.shape}")

    n_components = 2

    u = umap.UMAP(n_components=n_components, min_dist=min_dist, n_neighbors=n_neighbors, metric='cosine').fit_transform(
        query_data)  # reduces from 32 to 2
    print(u.shape)

    fig = plt.figure()

    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1], c = color_4_umap)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    plt.title('UMAP')
    plt.savefig("./results/jupytest/siamese.png")
