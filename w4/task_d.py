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

    with torch.no_grad():
        for ii, (img, _) in enumerate(query):
            query_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()

    print(f"QUERY SHAPE {query_data.shape}")
    embedding = umap.UMAP(n_components=2, min_dist = min_dist,n_neighbors = n_neighbors, metric='hellinger').fit(query_data) # reduces from 32 to 2
    print(f"EMBEDING SHAPE {embedding.shape}")

    f = umap.plot.points(embedding,
                         labels=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street',
                                 'tallbuilding'])

    umap.plot.show(f)
    plt.imsave("/results/jupytest/siamese.png")
