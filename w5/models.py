import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torch import optim

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, embed_size):
        super().__init__()
        self.linear = nn.Linear(dim, embed_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.squeeze(-1).squeeze(-1)
        x = self.activation(x)
        x = self.linear(x)
        return x


class Triplet(nn.Module):
    def __init__(self, dim1, dim2, embed_size):
        super(Triplet).__init__()
        self.image_model = EmbeddingLayer(dim1,embed_size)
        self.text_model = EmbeddingLayer(dim2,embed_size)
    
    def forward(self, x1,x2,x3):
        out1 = self.image_model(x1)
        out2 = self.text_model(x2)
        out3 = self.text_model(x3)
        return out1, out2, out3

class TripletLossModel:
    def __init__(self, dim1, dim2, embed_size, lr, loss_func):
        self.image_encoder = EmbeddingLayer(dim1,embed_size)
        self.text_encoder = EmbeddingLayer(dim2,embed_size)
        self.loss_func = loss_func
        params = list(self.text_encoder.parameters())
        self.params = params
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=5e-4)
    
    def cuda(self):
        """switch cuda
        """
        self.image_encoder.cuda()
        self.text_encoder.cuda()

    def cpu(self):
        """switch cpu
        """
        self.image_encoder.cpu()
        self.text_encoder.cpu()

    def state_dict(self):
        state_dict = [self.image_encoder.state_dict(), self.text_encoder.state_dict()]
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.image_encoder.load_state_dict(state_dict[0])
        self.text_encoder.load_state_dict(state_dict[1])
    
    def train_start(self):
        """switch to train mode
        """
        self.image_encoder.train()
        self.text_encoder.train()
    
    def val_start(self):
        """switch to evaluate mode
        """
        self.image_encoder.eval()
        self.text_encoder.eval()
    
    def forward(self, image_triple, caption_triple):
        image, pos_cap, neg_cap = image_triple.get_batch()
        image_encoded = self.image_encoder(image)
        pos_text_encoded = self.text_encoder(pos_cap[0])
        neg_text_encoded = self.text_encoder(neg_cap[0])
        loss = self.loss_func(image_encoded, pos_text_encoded, neg_text_encoded)

        # execute caption_triple
        caption, pos_img, neg_img = caption_triple.get_batch()
        text_encoded = self.text_encoder(caption[0])
        pos_image_encoded = self.image_encoder(pos_img)
        neg_image_encoded = self.image_encoder(neg_img)
        caption_triple_loss = self.loss_func(text_encoded, pos_image_encoded, neg_image_encoded)
        
        #loss = image_triple_loss + caption_triple_loss

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        # compute gradient and do SGD step
        loss.backward()
        self.optimizer.step()

        return loss.item()
    