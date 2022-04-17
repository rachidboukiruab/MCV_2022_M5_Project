import torch


class ImgEncoder(torch.nn.Module):
    def __init__(self, embedding_size = 1000):
        super(ImgEncoder, self).__init__()

        self.linear1 = torch.nn.Linear(4096, embedding_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


class TextEncoder(torch.nn.Module):
    # FIXME: text encoder equal size for every embed
    def __init__(self, embedding_size = 1000):
        super(TextEncoder, self).__init__()
        self.linear1 = torch.nn.Linear(300, embedding_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x
