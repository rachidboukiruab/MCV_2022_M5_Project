from pytorch_metric_learning import miners, losses
from torch import optim
from torch.utils.data import DataLoader

from snd_utils import MyDataset

img_features_file = '/home/group01/mcv/datasets/Flickr30k/vgg_feats.mat'
text_features_file = '/home/group01/mcv/datasets/Flickr30k/fasttext_feats.npy'
miner = miners.MultiSimilarityMiner()
loss_func = losses.TripletMarginLoss()

optimizer = optim.Adam(list(imageNet.parameters())+list(textNet.parameters()), lr=args.lr, weight_decay=args.weight_decay)


train_set = MyDataset(img_features_file, text_features_file)
dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

# training loop
for i, (img_features, text_features, labels) in enumerate(dataloader):
    optimizer.zero_grad()
    labels = miner(img_features, text_features)
    loss = loss_func(img_features, text_features, labels)
    loss.backward()
    optimizer.step()
