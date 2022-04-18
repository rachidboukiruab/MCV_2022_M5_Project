from torch.nn import Module, Linear, ReLU
import numpy as np


class ImgEncoder(Module):
    def __init__(self, embedding_size = 1000):
        super(ImgEncoder, self).__init__()

        self.linear1 = Linear(4096, embedding_size)
        self.activation = ReLU()

        self.init_weights()

    def init_weights(self):
        # Linear
        r = np.sqrt(6.) / np.sqrt(self.linear1.in_features +
                                  self.linear1.out_features)
        self.linear1.weight.data.uniform_(-r, r)
        self.linear1.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


class TextEncoder(Module):
    # FIXME: text encoder equal size for every embed
    def __init__(self, embedding_size = 1000):
        super(TextEncoder, self).__init__()
        self.linear1 = Linear(300, embedding_size)
        self.activation = ReLU()

        self.init_weights()

    def init_weights(self):
        # Linear
        r = np.sqrt(6.) / np.sqrt(self.linear1.in_features +
                                  self.linear1.out_features)
        self.linear1.weight.data.uniform_(-r, r)
        self.linear1.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x
