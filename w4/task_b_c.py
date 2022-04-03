import matplotlib.pyplot as plt
import logging

from pathlib import Path

import pytorch_metric_learning
from pytorch_metric_learning import distances
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from pytorch_metric_learning import samplers
from pytorch_metric_learning import trainers
from pytorch_metric_learning import testers
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import umap
import torch
import torch.nn as nn
from cycler import cycler
from torch import optim

#from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import sys
import pickle

from models import create_headless_resnet18


""" def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show() """


def main(config):
    data_path = Path(config["data_path"])
    output_path = Path(config["out_path"])
    model = create_headless_resnet18(config["embed_size"])
    #model.load_state_dict(torch.load('/home/aharris/shared/m5/weights_contrastive_100_scheduler.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #summary(model)

    transfs = transforms.Compose([
        transforms.ColorJitter(brightness=.3, hue=.3),
        transforms.RandomResizedCrop(256, (0.15, 1.0)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(str(data_path / "train"), transform=transfs)
    test_dataset = ImageFolder(str(data_path / "test"), transform=test_transfs)

    data_labels = [x for _, x in dataset.samples]

    class_sampler = samplers.MPerClassSampler(
        labels=data_labels,
        m=config["batch_size"] // 8,
        batch_size=config["batch_size"],
        length_before_new_iter=len(dataset),
    )
    optimizer = optim.Adam(model.parameters(), 3E-4)

    model = model.to(device)

    if config["loss_type"] == "contrastive":
        loss_funcs = {
            "metric_loss": losses.ContrastiveLoss()
        }
        mining_funcs = {
            "tuple_miner": miners.PairMarginMiner()
        }
    else:   # Triplet loss
        loss_funcs = {
            "metric_loss": losses.TripletMarginLoss(margin=0.1)
        }
        mining_funcs = {
            #"tuple_miner": miners.MultiSimilarityMiner(epsilon=0.1)
            "tuple_miner" : miners.BatchHardMiner()
        }

    record_keeper, _, _ = logging_presets.get_record_keeper(
        str(output_path / "logs"), str(output_path / "tb")
    )
    #dataset_dict = {"val": test_dataset}
    model_folder = str(output_path / "models")
    hooks = logging_presets.get_hook_container(record_keeper)

    """ # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=umap.UMAP(),
        visualizer_hook=visualizer_hook,
        #dataloader_num_workers=1,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, test_interval=10, patience=1
    )"""

    metric_trainer = trainers.MetricLossOnly(
        models={"trunk": model},
        optimizers={"trunk_optimizer": optimizer},
        batch_size=config["batch_size"],
        loss_funcs=loss_funcs,
        mining_funcs=mining_funcs,  # {"subset_batch_miner": mining_func1, "tuple_miner": mining_func2}
        dataset=dataset,
        data_device=device,
        sampler=class_sampler,
        lr_schedulers= {"trunk":model, "step_type" : optim.lr_scheduler.StepLR(optimizer,)},
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=None,
    )
    metric_trainer.train(1, 100)   
    torch.save(model.state_dict(), '{}/weights_{}.pth'.format(config['out_path'], config['loss_type'])) 
    
    
    #feature extraction (embeddings):
    catalogue_meta = dataset.samples
    query_meta  = test_dataset.samples
    
    catalogue_data = np.empty((len(dataset), config['embed_size']))
    with torch.no_grad():
        for ii, (img, _) in enumerate(dataset):
            catalogue_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy() 

    with open("{}_catalogue_{}_{}.npy".format(config['feature_path'],config['loss_type'], config['embed_size']), "wb") as f:
        np.save(f, catalogue_data)

    query_data = np.empty((len(test_dataset), config['embed_size']))
    with torch.no_grad():
        for ii, (img, _) in enumerate(test_dataset):
            query_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy() 

    with open("{}_query_{}_{}.npy".format(config['feature_path'],config['loss_type'], config['embed_size']), "wb") as f:
        np.save(f, query_data)


    #Image retrieval:

    if config['retrieval_method'] == 'knn':
        catalogue_labels = np.asarray([x[1] for x in catalogue_meta])
        query_labels = np.asarray([x[1] for x in query_meta])

        knn = KNeighborsClassifier(n_neighbors=5)
        knn = knn.fit(catalogue_data, catalogue_labels)
        predictions = knn.predict(query_data)
        pr_prob = knn.predict_proba(query_data)
        neighbors = knn.kneighbors(query_data)[1]

        with open('./results/retrieval/knn_{}_{}_scheduler.pkl'.format(config['loss_type'], config['embed_size']),'wb') as handle:
            pickle.dump(neighbors,handle, protocol=pickle.HIGHEST_PROTOCOL)


    else: #euclidian top10

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

        outfile = open('./results/retrieval/top10_{}_{}_scheduler_hard.pkl'.format(config['loss_type'],config['embed_size']),'wb')
        pickle.dump(top10_results,outfile)
        outfile.close()



if __name__ == "__main__":
    config = {
        "data_path": "MIT",
        "out_path": "./results/jupytest/",
        "feature_path" : "./results/retrieval/",
        "retrieval_method" : "knn",
        "embed_size": 100,
        "batch_size": 128,
        "loss_type": "contrastive"
    }
    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)
    main(config)



