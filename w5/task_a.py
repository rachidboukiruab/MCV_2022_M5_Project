import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pickle
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models import ImgEncoder, TextEncoder
from utils import decay_learning_rate, mpk
from dataset import Img2TextDataset, FlickrImagesAndCaptions

import os

img_features_file = '/home/group01/mcv/datasets/Flickr30k/vgg_feats.mat'
text_features_file = '/home/group01/mcv/datasets/Flickr30k/fasttext_feats.npy'
dataset_path = '/home/group01/mcv/datasets/Flickr30k'
output_path = "./results/task_a/"
training = False

parser = ArgumentParser(
        description='Torch-based image classification system',
        formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument("anchor",
                    type=str,
                    help="Which modality to use as anchor")
parser.add_argument("num_epochs",
                    type=int,
                    help="Number of epochs")
parser.add_argument("lr",
                    type=float,
                    help="learning rate")
parser.add_argument("weight_decay",
                    type=float,
                    help="weight decay")
parser.add_argument("batch_size",
                    type=int,
                    help="batch size")
parser.add_argument("margin",
                    type=float,
                    help="change margin for triplet loss")
parser.add_argument("grad_clip",
                    type=float,
                    help="grad_clip")
parser.add_argument("gamma",
                    type=float,
                    help="Learning Rate Gamma")
parser.add_argument("mining_type",
                    type=str,
                    help="What mining modality to use (hard, semihard, easy, all)")

args = parser.parse_args()

def display_embeddings(text_embeddings, image_embeddings, text_labels, image_labels, epoch):
    reducer = umap.UMAP()
    n_text, _ = text_embeddings.shape
    n_imag, dim = image_embeddings.shape

    all_embeddings = np.vstack([text_embeddings, image_embeddings])
    all_embeddings = reducer.fit_transform(all_embeddings)

    text_embeddings = all_embeddings[:n_text]
    image_embeddings = all_embeddings[n_text:]

    plt.figure(dpi=300, figsize=(15, 15))
    plt.title(f"UMAP of the embedding space at epoch {epoch}")
    plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1], color="orange", label="Text Embeddings")
    plt.scatter(image_embeddings[:, 0], image_embeddings[:, 1], color="cyan", label="Image Embeddings")

    for ii, label in enumerate(text_labels):
        plt.annotate(label, (text_embeddings[ii, 0], text_embeddings[ii, 1]), color="red", alpha=0.5)
    for ii, label in enumerate(image_labels):
        plt.annotate(label, (image_embeddings[ii, 0], image_embeddings[ii, 1]), color="blue", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path + f"/plots/plot_epoch_{epoch}.png")
    plt.close()


def validate(valid_dataloader, image_model, text_model, anchor, epoch, file_name):
    all_img_features = []
    all_txt_features = []

    image_model.eval()
    text_model.eval()

    with torch.no_grad():
        for i, (img_features, txt_features) in enumerate(valid_dataloader):
            img_features = img_features.to(device)  # (batch, ifeatures)
            txt_features = txt_features.to(device)  # (batch, ncaptions, tfeatures)

            batch_size, ncaptions, tfeatures = txt_features.shape

            # Reshape textual features so they are all encoded at once
            txt_features = txt_features.reshape((-1, tfeatures))

            img_encoded = image_model(img_features)
            txt_encoded = text_model(txt_features)

            all_img_features.append(img_encoded.detach().to("cpu").numpy())
            all_txt_features.append(txt_encoded.detach().to("cpu").numpy())

    all_img_features = np.vstack(all_img_features)
    all_txt_features = np.vstack(all_txt_features)

    all_img_labels = np.arange(len(all_img_features))
    all_txt_labels = np.arange(len(all_img_features)).repeat(
        len(all_txt_features) // len(all_img_features)
    )

    knn = KNeighborsClassifier(5, metric="euclidean")

    if anchor == "text":
        knn = knn.fit(all_img_features, all_img_labels)
        neighbors = knn.kneighbors(all_txt_features, return_distance=False)
        predictions = all_img_labels[neighbors]

        p1 = mpk(all_txt_labels, predictions, 1)
        p5 = mpk(all_txt_labels, predictions, 5)
    else:
        knn = knn.fit(all_txt_features, all_txt_labels)
        neighbors = knn.kneighbors(all_img_features, return_distance=False)
        predictions = all_txt_labels[neighbors]

        p1 = mpk(all_img_labels, predictions, 1)
        p5 = mpk(all_img_labels, predictions, 5)

    if not training:
        all_features = {'txt_features': all_txt_features, 'img_features': all_img_features, 'txt_labels': all_txt_labels, 'img_labels': all_img_labels}
        #Save data
        with open('{}/{}_all_features_{}.pkl'.format(output_path, args.anchor, file_name), "wb") as f:
            pickle.dump(all_features, f)       
        print('Features and labels saved.')
    
    if not epoch % 100:
        display_embeddings(all_txt_features, all_img_features, all_txt_labels, all_img_labels, epoch)

    image_model.train()
    text_model.train()

    return p1, p5

if __name__ == '__main__':

    loss_func = nn.TripletMarginLoss(args.margin, p=2)

    train_set = FlickrImagesAndCaptions(dataset_path, "train")
    val_set = FlickrImagesAndCaptions(dataset_path, "val")
    test_set = FlickrImagesAndCaptions(dataset_path, "test")

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # TEXT & IMGS MODELS
    image_model = ImgEncoder()
    text_model = TextEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model.to(device)
    text_model.to(device)
    # init weights
    image_model.init_weights()
    text_model.init_weights()

    if training:
        # optimizer
        params = list(image_model.parameters())
        params += list(text_model.parameters())

        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        for epoch in range(args.num_epochs):
            decay_learning_rate(args.lr, optimizer, epoch)

            for i, img_triple in enumerate(train_dataloader):

                # execute image_triple
                img_features, pos_text_features, neg_text_features = img_triple
                img_features, pos_text_features, neg_text_features = img_features.to(
                    device), pos_text_features.to(device), neg_text_features.to(device)
                image_encoded = image_model(img_features)
                pos_text_encoded = text_model(pos_text_features)
                neg_text_encoded = text_model(neg_text_features)

                loss = loss_func(image_encoded, pos_text_encoded, neg_text_encoded)

                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    clip_grad_norm_(params, args.grad_clip)
                optimizer.step()

                print(f'epoch: {epoch}\titeration: {i}\tLoss: {loss}')
        
        state_dict = [image_model.state_dict(), text_model.state_dict()]
        model_folder = str(output_path + "/models")
        os.makedirs(model_folder, exist_ok=True)
        torch.save(state_dict, '{0}/Image2Text_weights.pth'.format(model_folder))
    
    else:
        #LOAD PRETRAINED WEIGHTS
        state_dict =  torch.load('{}/models/Image2Text_weights.pth'.format(output_path,args.anchor))
        image_model.load_state_dict(state_dict[0])
        text_model.load_state_dict(state_dict[1])

        # optimizer
        params = list(image_model.parameters())
        params += list(text_model.parameters())

        optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ExponentialLR(optimizer, args.gamma)

        image_model.train()
        text_model.train()

        p1, p5 = validate(train_dataloader, image_model, text_model, args.anchor, -1, 'train')
        p1, p5 = validate(val_dataloader, image_model, text_model, args.anchor, -1, 'validation')
        p1, p5 = validate(test_dataloader, image_model, text_model, args.anchor, -1, 'test')
