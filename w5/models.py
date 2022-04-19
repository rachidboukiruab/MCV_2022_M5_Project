from torch.nn import Module, Linear, ReLU, init,Sequential
import numpy as np
from torchvision import models



class ImgEncoder(Module):
    def __init__(self, dim=4096,embedding_size = 1000):
        super(ImgEncoder, self).__init__()

        self.linear1 = Linear(dim, embedding_size)
        self.activation = ReLU()
        self.init_weights()

    def init_weights(self):
        # Linear
        init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


class TextEncoder(Module):
    def __init__(self, embedding_size = 1000):
        super(TextEncoder, self).__init__()
        self.linear1 = Linear(300, embedding_size)
        self.activation = ReLU()

        self.init_weights()

    def init_weights(self):
        # Linear
        init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x

def get_faster():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    features = list(model.roi_heads.children())[:-1]
    model.roi_heads = Sequential(*features)
    return model
