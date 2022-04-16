from pytorch_metric_learning import miners, losses
from torch import optim
from torch.utils.data import DataLoader

from snd_utils import Img2TextDataset, decay_learning_rate

img_features_file = '/home/group01/mcv/datasets/Flickr30k/vgg_feats.mat'
text_features_file = '/home/group01/mcv/datasets/Flickr30k/fasttext_feats.npy'

loss_func = losses.TripletMarginLoss()


train_set = Img2TextDataset(img_features_file, text_features_file)
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

# training loop
for epoch in range(12):
    decay_learning_rate(init_lr, optimizer, epoch)

    for i, (img_triple, caption_triple) in enumerate(train_dataloader):

        # execute image_triple
        img_features, pos_text_features, neg_text_features = img_triple
        image_triple_loss = loss_func(img_features, pos_text_features, neg_text_features)

        # execute caption_triple
        caption, pos_img, neg_img = caption_triple
        caption_triple_loss = loss_func(caption, pos_img, neg_img)

        loss = image_triple_loss + caption_triple_loss
        optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        print(f'epoch: {epoch}\titeration: {i}\tLoss: {loss}')