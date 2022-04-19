import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models import ImgEncoder, TextEncoder
from utils import decay_learning_rate
from dataset import Text2ImgDataset
import os

img_features_file = '/home/group01/mcv/datasets/Flickr30k/vgg_feats.mat'
text_features_file = '/home/group01/mcv/datasets/Flickr30k/fasttext_feats.npy'
output_path = "./results/task_b/"

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
parser.add_argument("margin",
                    type=float,
                    help="change margin for triplet loss")
parser.add_argument("grad_clip",
                    type=int,
                    help="grad_clip")

args = parser.parse_args()

if __name__ == '__main__':

    loss_func = nn.TripletMarginLoss(args.margin, p=2)

    train_set = Text2ImgDataset(img_features_file, text_features_file)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # TEXT & IMGS MODELS
    image_model = ImgEncoder()
    text_model = TextEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model.to(device)
    text_model.to(device)
    # init weights
    image_model.init_weights()
    text_model.init_weights()

    # optimizer
    params = list(image_model.parameters())
    params += list(text_model.parameters())

    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    for epoch in range(args.num_epochs):
        decay_learning_rate(args.lr, optimizer, epoch)

        for i, caption_triple in enumerate(train_dataloader):

            # execute caption_triple
            caption, pos_img, neg_img = caption_triple
            caption, pos_img, neg_img = caption.to(device), pos_img.to(device), neg_img.to(device)
            caption_encoded = text_model(caption)
            pos_img_encoded = image_model(pos_img)
            neg_img_encoded = image_model(neg_img)
            loss = loss_func(caption_encoded, pos_img_encoded, neg_img_encoded)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            print(f'epoch: {epoch}\titeration: {i}\tLoss: {loss}')
    
    state_dict = [image_model.state_dict(), text_model.state_dict()]
    model_folder = str(output_path + "/models")
    os.makedirs(model_folder, exist_ok=True)
    torch.save(state_dict, '{0}/Text2Image_weights.pth'.format(model_folder))
