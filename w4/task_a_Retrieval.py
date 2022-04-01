import pickle
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, average_precision_score
#from ml_metrics import mapk
import torch
import os

data_path = Path("/home/group01/mcv/datasets/MIT_split")
feature_path = Path("./results/retrieval")
os.makedirs(feature_path, exist_ok=True)

'''def compute_mapk(gt,hypo,k_val):
    apk_list = []
    for ii,query in enumerate(gt):
        for jj,sq in enumerate(query):
            apk_val = 0.0
            if len(hypo[ii]) > jj:
                apk_val = apk([sq],hypo[ii][jj], k_val)
            apk_list.append(apk_val)
            
    return np.mean(apk_list)'''

def evaluation_mapk(actual,predicted):
    k_1=mapk(actual, predicted, k=1)
    k_5=mapk(actual, predicted, k=5)
    return (k_1,k_5)

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

    '''knn = KNeighborsClassifier(n_neighbors=5)
    knn = knn.fit(catalogue_data, catalogue_labels)
    predictions = knn.predict(query_data)
    pr_prob = knn.predict_proba(query_data)

    one_hot = np.zeros((predictions.shape[0], max(predictions) + 1), dtype=int)
    one_hot[predictions] = 1

    f1 = f1_score(query_labels, predictions, average="macro")
    ap = average_precision_score(one_hot, pr_prob)'''
    results = []
    top10_results = []
    score = {}
    print("Searching...")
    for i in range(len(query_data)):
        query_img = query_meta[i]
        query_feature = query_data[i]
        query_feature = np.array(query_feature)
        query_feature = torch.from_numpy(query_feature)
        for j in range(len(catalogue_data)):
            # if database contains query-image
            #img = catalogue_meta[j][0]
            # print("origin: "+img)
            catalogue_feature = catalogue_data[j]
            catalogue_feature = np.array(catalogue_feature)
            catalogue_feature = torch.from_numpy(catalogue_feature)
            output = torch.dist(catalogue_feature, query_feature, p=1)
            if output == 0:
                continue
            # print(output)
            score[j] = abs(output)
        print("Search Finished")
        #top10 = sorted(score.items(), key=lambda score: score[1], reverse=False)[:10]
        top10 = sorted(score, key=score.get, reverse=False)[:10]
        print("10 most relevant pictures for the query image ", query_img[0],"from class", query_img[1], " are as below: ")
        results_class = [catalogue_meta[j][1] for j in top10]
        #names = [catalogue_meta[j][0] for j in top10]
        print("(img_name, class)")
        for index in top10:
            print(catalogue_meta[index])
            # image = plt.imread(img)
            # plt.imshow(image)
            # plt.show()
            # plt.savefig('./result/picture')
        print(results_class)
        results.append(results_class)
        top10_results.append(top10)
    
    ''' k_1, k_5 = evaluation_mapk(list(query_labels),results)
    print('MAP@1=',k_1)
    print('MAP@5=',k_5)'''
    outfile = open(feature_path / "top10_results.pkl",'wb')
    pickle.dump(top10_results,outfile)
    outfile.close()
        

    #print("AP = ", ap)
    #print("F1 = ", f1)