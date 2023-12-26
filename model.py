import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=12, stride=3, padding=(6, 7))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=9, stride=3, padding=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=6, stride=3, padding=(3, 4))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=3, padding=(2, 2))

        # second Layers
        self.conv5 = nn.Sequential(nn.ZeroPad2d(1),
                                   nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
                                   nn.MaxPool2d(kernel_size=2))
        self.bm_1 = nn.BatchNorm2d(32)
        self.conv6 = nn.Sequential(nn.ZeroPad2d(1),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
                                   nn.MaxPool2d(kernel_size=2))
        self.bm_2 = nn.BatchNorm2d(64, affine=False)
        self.conv7 = nn.Sequential(nn.ZeroPad2d(1),
                                   nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
                                   nn.MaxPool2d(kernel_size=2))
        self.bm_3 = nn.BatchNorm2d(128, affine=False)
        self.conv8 = nn.Sequential(nn.ZeroPad2d(1),
                                   nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0),
                                   nn.MaxPool2d(kernel_size=2))

        # mlp to classifier
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(in_features=1024, out_features=512)
        self.l2 = nn.Linear(in_features=512, out_features=256)
        self.l3 = nn.Linear(in_features=256, out_features=100)

        # classifier
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # preparation
        x = x.to(torch.float32)
        batch_s = x.size()[0]
        # print(x.size())
        x = x.resize_(batch_s, 1, 1920, 1920)

        # feature extraction
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # pattern extraction
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.conv5(y)
        y = self.bm_1(y)
        y = self.conv6(y)
        y = self.bm_2(y)
        y = self.conv7(y)
        y = self.bm_3(y)
        y = self.conv8(y)

        # flatten & MLP
        y = self.flat(y)
        y = self.flat(y)
        y = F.relu(self.l1(y))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        # Output Layer
        y = torch.sigmoid(self.fc2(y))

        return y.reshape(batch_s)
