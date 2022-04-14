import numpy as np
from scipy.io import loadmat
from utils import mpk, mAP
import json
from sklearn.neighbors import KNeighborsClassifier
import pickle



def main(config):
    data_path = config['data_path']
    out_path = config['data_path']
    with open(f'{data_path}/train.json') as f: #CATALOGUE META: select sentences id from training set
        train = json.load(f) 

    with open(f'{data_path}/test.json') as f: #QUERY META: images from test set
        test = json.load(f)
    catalogue_meta = [(train[i]['filename'],train[i]['sentids']) for i in range(len(train))]
    query_meta = [(test[i]['filename'],test[i]['sentids']) for i in range(len(test))]

    if config['type'] == 'image2text':

        catalogue_data = np.load(f'{data_path}/fasttext_feats.npy', allow_pickle=True)
    
        query_data = loadmat(f'{data_path}/vgg_feats.mat')

    else:
        quey_data = np.load(f'{data_path}/fasttext_feats.npy', allow_pickle=True)
    
        catalogue_data = loadmat(f'{data_path}/vgg_feats.mat')

    ############REVISAR RETRIEVAL (5 POSIBLES SENTENCES)
    catalogue_labels = np.asarray([x[1] for x in catalogue_meta])
    query_labels = np.asarray([x[1] for x in query_meta])

    # Image retrieval:

    knn = KNeighborsClassifier(n_neighbors=len(catalogue_labels), p=1)
    knn = knn.fit(catalogue_data, catalogue_labels)
    neighbors = knn.kneighbors(query_data)[1]
    # print(neighbors)

    neighbors_labels = []
    for i in range(len(neighbors)):
        neighbors_class = [catalogue_meta[j][1] for j in neighbors[i]]
        neighbors_labels.append(neighbors_class)

    query_labels = [x[1] for x in query_meta]

    p_1 = mpk(query_labels, neighbors_labels, 1)
    p_5 = mpk(query_labels, neighbors_labels, 5)
    print('P@1=', p_1)
    print('P@5=', p_5)

    map = mAP(query_labels, neighbors_labels)
    print('mAP=', map)
    with open(f'{out_path}/image2text.pkl', 'wb') as handle:
        pickle.dump(neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    config = {
        "data_path": "/home/aharris/shared/m5/Flickr30k",
        "out_path": "./results/",
        "type": "image2text"
    }
    main(config)