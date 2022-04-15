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
from scipy.io import loadmat

import numpy as np
import umap
import torch
import torch.nn as nn
from cycler import cycler
from torch import optim

# from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder


from models import Triplet



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

    img_features = loadmat(f'{data_path}/vgg_feats.mat')
    txt_features = np.load(f'{data_path}/fasttext_feats.npy', allow_pickle=True)

    img_dimensions = np.asarray(img_features).shape
    txt_dimensions = txt_features.shape
    
    model = Triplet(img_dimensions[0], txt_dimensions[2],config["embed_size"])
    #model.load_state_dict(torch.load('/home/aharris/shared/m5/CONTRASTIVE.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # summary(model)

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

    # Triplet loss
    loss_funcs = {
        "metric_loss": losses.TripletMarginLoss(margin=0.1)
    }
    mining_funcs = {
        # "tuple_miner": miners.MultiSimilarityMiner(epsilon=0.1)
        """ "tuple_miner" : miners.BatchEasyHardMiner(
                            pos_strategy=miners.BatchEasyHardMiner.EASY,
                            neg_strategy=miners.BatchEasyHardMiner.SEMIHARD,
                            allowed_pos_range=None,
                            allowed_neg_range=None,
                            ) """
        "tuple_miner": miners.BatchHardMiner()
    }

    record_keeper, _, _ = logging_presets.get_record_keeper(
        str(output_path / "logs"), str(output_path / "tb")
    )
    dataset_dict = {"val": test_dataset}
    model_folder = str(output_path / "models")
    hooks = logging_presets.get_hook_container(record_keeper)

    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=umap.UMAP(),
        #visualizer_hook=visualizer_hook,
        dataloader_num_workers=1,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, test_interval=10, patience=1
    )

    metric_trainer = trainers.MetricLossOnly(
        models={"trunk": model},
        optimizers={"trunk_optimizer": optimizer},
        batch_size=config["batch_size"],
        loss_funcs=loss_funcs,
        mining_funcs=mining_funcs,  # {"subset_batch_miner": mining_func1, "tuple_miner": mining_func2}
        dataset=dataset,
        data_device=device,
        sampler=class_sampler,
        lr_schedulers={"trunk": model, "step_type": optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)},
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )

    metric_trainer.train(1, 100)
    


if __name__ == "__main__":
    config = {
        "data_path": "/home/aharris/shared/m5/Flickr30k",
        "out_path": "./results/jupytest/",
        "feature_path": "./results/retrieval/",
        "embed_size": 32,
        "batch_size": 64,
    }
    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)
    main(config)
