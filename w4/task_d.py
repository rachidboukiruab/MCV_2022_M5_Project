from pathlib import Path

import torch
import umap
import umap.plot

import numpy as np
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

    model = create_headless_resnet18()
    # LOAD PRE_TRAINED WEIGHTS
    model.load_state_dict(torch.load(trained_path / weights_filename[0]))

    query = ImageFolder(str(data_path / "test"))
    query_data = np.empty((len(query), 32))

    with torch.no_grad():
        for ii, (img, _) in enumerate(query):
            query_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()

    embedding = umap.UMAP(n_components=2, min_dist = min_dist,n_neighbors = n_neighbors, metric='hellinger').fit(query) # reduces from 32 to 2
    print(f"EMBEDING SHAPE {embedding.shape}")

    f = umap.plot.points(embedding,
                         labels=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street',
                                 'tallbuilding'])

