import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

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
    