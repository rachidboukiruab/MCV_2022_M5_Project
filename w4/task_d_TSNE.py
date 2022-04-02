from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

    query = ImageFolder(str(data_path / "train"), transform=transfs)
    query_data = np.empty((len(query), EMBED_SHAPE))

    color_4_umap = list()
    select_color = ['#8db6f7', '#b98df7', '#f78df2', '#f78da8', '#f7a68d', '#f7e08d',
                    '#bff78d', '#8df7af']
    with torch.no_grad():
        for ii, (img, label) in enumerate(query):
            query_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()
            color_4_umap.append(select_color[label])

    print(f"QUERY LEN {len(query_data)}")

    pca = PCA(n_components=EMBED_SHAPE)
    pca.fit(query_data)
    query_features_compressed = pca.transform(query_data)

    tsne_results = TSNE(n_components=2, verbose=1, metric='euclidean').fit_transform(query_features_compressed)

    colormap = plt.cm.get_cmap('coolwarm')

    scatter_plot = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=color_4_umap, cmap=colormap)
    plt.colorbar(scatter_plot)
    plt.show()

    plt.savefig("./results/jupytest/tsne_siamese.png")

