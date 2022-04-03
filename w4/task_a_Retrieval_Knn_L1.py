import pickle
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
from utils import mpk, mAP

#data_path = Path("/home/group01/mcv/datasets/MIT_split")
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

    knn = KNeighborsClassifier(n_neighbors=5, metric = "manhattan")
    knn = knn.fit(catalogue_data, catalogue_labels)
    neighbors = knn.kneighbors(query_data, return_distance=False)
    #print(neighbors)

    outfile = open(feature_path / "KnnL1_results_new.pkl",'wb')
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