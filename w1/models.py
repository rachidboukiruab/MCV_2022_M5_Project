import torch
from torch import nn


class SmallNet(nn.Module):
    def __init__(self, nclasses):
        super(SmallNet, self).__init__()

        self.nclasses = nclasses

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=2)
        self.gmap = nn.MaxPool2d(kernel_size=8)

        self.conv1 = nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding="same")
        self.conv2 = nn.Conv2d(64, 24, (3, 3), stride=(1, 1), padding="same")
        self.conv3 = nn.Conv2d(24, 48, (3, 3), stride=(1, 1), padding="same")
        self.conv4 = nn.Conv2d(48, 96, (3, 3), stride=(1, 1), padding="same")
        self.conv5 = nn.Conv2d(96, 192, (3, 3), stride=(1, 1), padding="same")
        self.conv6 = nn.Conv2d(192, 192, (3, 3), stride=(1, 1), padding="same")
        self.conv7 = nn.Conv2d(192, 384, (3, 3), stride=(1, 1), padding="same")
        self.linear = nn.Linear(384, self.nclasses)

    def forward(self, x):
        x = self.conv1(x)       # Batch x 3 x 256 x 256
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv2(x)       # Batch x 64 x 128 x 128
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv3(x)       # Batch x 24 x 64 x 64
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv4(x)       # Batch x 48 x 32 x 32
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv5(x)       # Batch x 96 x 16 x 16
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv7(x)       # Batch x 192 x 8 x 8
        x = self.relu(x)

        x = self.gmap(x)        # Batch x 384 x 8 x 8
        x = torch.squeeze(x)    # Batch x 384 x 1 x 1
        x = self.linear(x)      # Batch x 384 -> Batch x Classes

        return x  # UNNORMALISED LOGITS! CAREFUL! (to use w/ cross entropy loss)

class Team3Model(nn.Module):
    def __init__(self, nclasses):
        super(Team3Model, self).__init__()

        self.nclasses = nclasses

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn. Conv2d(16, 32, 3)
        self.conv3 = nn. Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.Avgpool = nn.AvgPool2d(64)
        self.linear = nn.Linear(64, self.nclasses)       
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.Avgpool(nn.functional.relu(self.conv3(x)))
        x = torch.squeeze(x)
        x = nn.functional.softmax(self.linear(x))
        

        return x  # UNNORMALISED LOGITS! CAREFUL! (to use w/ cross entropy loss)
