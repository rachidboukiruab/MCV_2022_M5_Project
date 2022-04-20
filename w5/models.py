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



class FasterRCNN(Module):
            def __init__(self):
                super(FasterRCNN, self).__init__()
                self.original_model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
                self.fc = Linear(2048,1024)
                self.features = Sequential(*list(self.original_model.backbone.children())[:-1], self.fc) 


            def forward(self, x):
                out = self.features(x)
                return out




    
