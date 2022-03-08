from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, nclasses):
        super(SimpleNet, self).__init__()

        self.nclasses = nclasses

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=2)
        self.gmap = nn.MaxPool2d(kernel_size=8)
        #self.sigmoid =

        self.conv1 = nn.Conv2d(3  ,  64, 3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(64 ,  24, 3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(24 ,  48, 3, stride=1, padding="same")
        self.conv4 = nn.Conv2d(48 ,  96, 3, stride=1, padding="same")
        self.conv5 = nn.Conv2d(96 , 192, 3, stride=1, padding="same")
        self.conv6 = nn.Conv2d(192, 192, 3, stride=1, padding="same")
        self.conv7 = nn.Conv2d(192, 384, 3, stride=1, padding="same")
        self.linear = nn.Linear(384, self.nclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.conv7(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.gmap(x)
        x = self.linear(x)

        print(x)

        return x # UNNORMALISED LOGITS! CAREFUL! (to use w/ cross entropy loss)
