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
    min_dist = 0.1
    n_neighbors = 50

    #########
    data_path = Path("/home/group01/mcv/datasets/MIT_split")
    feature_path = Path("./results/retrieval")
    trained_path = Path("./results/jupytest")

    weights_filename = "weights_contrastive.pth"

    EMBED_SHAPE = 64

    model = create_headless_resnet18(EMBED_SHAPE)
    # LOAD PRE_TRAINED WEIGHTS
    model.load_state_dict(torch.load(trained_path / weights_filename))

    transfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    query = ImageFolder(str(data_path / "test"), transform=transfs)
    query_data = np.empty((len(query), EMBED_SHAPE))

    color_4_umap = list()
    select_color = ['#8db6f7', '#b98df7', '#f78df2', '#f78da8', '#f7a68d', '#f7e08d',
                    '#bff78d', '#8df7af']
    with torch.no_grad():
        for ii, (img, label) in enumerate(query):
            query_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()
            color_4_umap.append(select_color[label])

    print(f"QUERY SHAPE {query_data.shape}")

    # 2 for 2D, 3 for 3D
    n_components = 3

    u = umap.UMAP(n_components=n_components, min_dist=min_dist, n_neighbors=n_neighbors, metric='manhattan').fit_transform(
        query_data)  # reduces from 32 to 2
    print(u.shape)

    fig = plt.figure()

    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1], c=color_4_umap)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=color_4_umap)
    plt.title('UMAP')
    plt.savefig("./results/jupytest/siamese.png")
