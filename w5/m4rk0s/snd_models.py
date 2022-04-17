import torch


class ImgEncoder(torch.nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()

        self.linear1 = torch.nn.Linear(4096, 1000)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


class TextEncoder(torch.nn.Module):
    # FIXME: text encoder equal size for every embed
    def __init__(self):
        super(TextEncoder, self).__init__()

    def forward(self, x):
        x = x.flatten()
        linear1 = torch.nn.Linear(x.shape[0], 1000)  # esto daría cada vez una red distinta ??
        activation = torch.nn.ReLU()
        x = linear1(x)
        x = activation(x)
        return x
