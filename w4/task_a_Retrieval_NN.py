import pickle
from pathlib import Path
import numpy as np
from utils import mpk, mAP
import torch
import os

data_path = Path("/home/group01/mcv/datasets/MIT_split")
feature_path = Path("./results/retrieval")
os.makedirs(feature_path, exist_ok=True)


if __name__ == '__main__':
    #Load the features/embeding data obtained from training set
    with open(feature_path / "queries.npy", "rb") as f:
        query_data = np.load(f)
    with open(feature_path / "catalogue.npy", "rb") as f:
        catalogue_data = np.load(f)

    #Load the features/embeding data obtained from test set
    with open(feature_path / "query_meta.pkl", "rb") as f:
        query_meta = pickle.load(f)
    with open(feature_path / "catalogue_meta.pkl", "rb") as f:
        catalogue_meta = pickle.load(f)

    
    catalogue_labels = np.asarray([x[1] for x in catalogue_meta])
    query_labels = np.asarray([x[1] for x in query_meta])

    neighbors = []
    score = {}
    print("Searching...")
    for i in range(len(query_data)):
        query_img = query_meta[i]
        query_feature = query_data[i]
        query_feature = np.array(query_feature)
        query_feature = torch.from_numpy(query_feature)
        for j in range(len(catalogue_data)):
            catalogue_feature = catalogue_data[j]
            catalogue_feature = np.array(catalogue_feature)
            catalogue_feature = torch.from_numpy(catalogue_feature)
            output = torch.dist(catalogue_feature, query_feature, p=1)
            score[j] = abs(output)
        print("Search Finished")
        sorted_imgs = sorted(score, key=score.get, reverse=False)
        neighbors.append(sorted_imgs)
    
    outfile = open(feature_path / "NN_results.pkl",'wb')
    pickle.dump(neighbors,outfile)
    outfile.close()
        

    neighbors_labels = []
    for i in range(len(neighbors)):
        neighbors_class = [catalogue_meta[j][1] for j in neighbors[i]]
        neighbors_labels.append(neighbors_class)

    query_labels = [x[1] for x in query_meta]

    p_1 = mpk(query_labels,neighbors_labels, 1)
    p_5 = mpk(query_labels,neighbors_labels, 5)
    print('P@1=',p_1)
    print('P@5=',p_5)

    map = mAP(query_labels,neighbors_labels)
    print('mAP=',map)