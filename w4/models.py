import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 61 * 61, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.linear = nn.Linear(512, embed_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.squeeze(-1).squeeze(-1)
        x = self.activation(x)
        x = self.linear(x)
        return x


class FlattenOnly(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)


def create_headless_resnet18(embed_size):
    embed = EmbeddingLayer(embed_size)
    model = models.resnet18(pretrained=True, progress=False)
    model = nn.Sequential(*list(model.children())[:-1], embed)
    return model


def create_headless_resnet18_noembed():
    flatten = FlattenOnly()
    model = models.resnet18(pretrained=True, progress=False)
    model = nn.Sequential(*list(model.children())[:-1], flatten)
    return model
