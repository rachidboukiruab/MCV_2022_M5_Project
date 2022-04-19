import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models import ImgEncoder, TextEncoder, FasterRCNN
from utils import decay_learning_rate
from dataset import FlickrFaster
import os
import numpy as np
import sys
from torchvision import transforms
from PIL import Image




def extract_visual_features(model,device, filepath,in_features):
    model.to(device=device)

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dir = os.listdir(filepath)
    image_features = np.empty((in_features,len(dir)))
    
    for i in dir: 
        image = tfms(Image.open(os.path.join(filepath,i)))
        model.eval()
        with torch.no_grad():
            img_tensor = image.to(device=device)
            output = model(img_tensor.unsqueeze(0))
            print(output)
            print(len(output))
            break
            #image_features[i,:] = model(img_tensor.unsqueeze(0)).numpy

            break
    return image_features



text_features_file = '/home/aharris/shared/m5/Flickr30k/fasttext_feats.npy'
train_path = '/home/aharris/shared/m5/Flickr30k/train/'
output_path = "./results/task_a/"

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    faster = FasterRCNN()
    print(faster)
    
    
    img_features = extract_visual_features(faster,device,train_path,in_features=2048)
    sys.exit()

    loss_func = nn.TripletMarginLoss(args.margin, p=2)

    train_set = FlickrFaster(img_features, text_features_file)
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # TEXT & IMGS MODELS
    image_model = ImgEncoder(dim=2048)
    text_model = TextEncoder()
    
    image_model.to(device)
    text_model.to(device)

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
