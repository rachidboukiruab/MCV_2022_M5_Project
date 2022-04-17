import torch
from pytorch_metric_learning import losses
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from snd_models import ImgEncoder, TextEncoder
from snd_utils import Img2TextDataset, decay_learning_rate

img_features_file = '/home/group01/mcv/datasets/Flickr30k/vgg_feats.mat'
text_features_file = '/home/group01/mcv/datasets/Flickr30k/fasttext_feats.npy'
num_epochs = 12

loss_func = losses.TripletMarginLoss()

train_set = Img2TextDataset(img_features_file, text_features_file)
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
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

optimizer = Adam(params, lr=1e-2, weight_decay=5e-4)

# training loop
for epoch in range(num_epochs):
    decay_learning_rate(1e-2, optimizer, epoch)

    for i, (img_triple, caption_triple) in enumerate(train_dataloader):

        # execute image_triple
        img_features, pos_text_features, neg_text_features = img_triple
        img_features, pos_text_features, neg_text_features = img_features.to(
            device), pos_text_features.to(device), neg_text_features.to(device)
        image_encoded = image_model(img_features)
        pos_text_encoded = text_model(pos_text_features)
        neg_text_encoded = text_model(neg_text_features)
        print(image_encoded.shape, pos_text_encoded.shape, neg_text_encoded.shape)
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
