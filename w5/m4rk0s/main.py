import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from snd_models import ImgEncoder, TextEncoder
from snd_utils import Img2TextDataset, decay_learning_rate

img_features_file = '/home/group01/mcv/datasets/Flickr30k/vgg_feats.mat'
text_features_file = '/home/group01/mcv/datasets/Flickr30k/fasttext_feats.npy'

parser = ArgumentParser(
        description='Torch-based image classification system',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
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

args = parser.parse_args()

if __name__ == '__main__':

    loss_func = nn.TripletMarginLoss(0.7, p=2)

    train_set = Img2TextDataset(img_features_file, text_features_file)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    grad_clip = 2

    # TEXT & IMGS MODELS
    image_model = ImgEncoder()
    text_model = TextEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model.to(device)
    text_model.to(device)

    # optimizer
    params = list(image_model.parameters())
    params += list(text_model.parameters())

    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    for epoch in range(args.num_epochs):
        decay_learning_rate(args.lr, optimizer, epoch)

        for i, (img_triple, caption_triple) in enumerate(train_dataloader):

            # execute image_triple
            img_features, pos_text_features, neg_text_features = img_triple
            img_features, pos_text_features, neg_text_features = img_features.to(
                device), pos_text_features.to(device), neg_text_features.to(device)
            image_encoded = image_model(img_features)
            pos_text_encoded = text_model(pos_text_features)
            neg_text_encoded = text_model(neg_text_features)

            image_triple_loss = loss_func(image_encoded, pos_text_encoded, neg_text_encoded)

            # execute caption_triple
            caption, pos_img, neg_img = caption_triple
            caption, pos_img, neg_img = caption.to(device), pos_img.to(device), neg_img.to(device)
            caption_encoded = text_model(caption)
            pos_img_encoded = image_model(pos_img)
            neg_img_encoded = image_model(neg_img)
            caption_triple_loss = loss_func(caption_encoded, pos_img_encoded, neg_img_encoded)

            loss = image_triple_loss + caption_triple_loss
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(params, grad_clip)
            optimizer.step()

            print(f'epoch: {epoch}\titeration: {i}\tLoss: {loss}')
