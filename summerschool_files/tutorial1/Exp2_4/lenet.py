import torch 
import torch.nn as nn
from initialize import weights_init_kaiming_normal

class LeNet5(nn.Module):
    # define LetNet5 
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.acti1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.acti2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        self.fc3 = nn.Linear(in_features=400,out_features=120)
        self.acti3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=120,out_features=84)
        self.acti4 = nn.ReLU()
        self.fc5 = nn.Linear(in_features=84,out_features=num_classes)

        weights_init_kaiming_normal(self)

    def forward(self, x):
        # the first convolutional, activation and maxpooling layer
        x = self.conv1(x)
        x = self.acti1(x)
        x = self.maxpool1(x)

        # the second convolutional, activation and maxpooling layer
        x = self.conv2(x)
        x = self.acti2(x)
        x = self.maxpool2(x)

        # stack the activation maps into 1d vector
        x = x.view(-1, 400)

        # third fully-connected (fc) layer and activation layer 
        x = self.fc3(x)
        x = self.acti3(x)

        # fourth fully-connected layer and activation layer
        x = self.fc4(x)
        x = self.acti4(x)

        # last fc layer
        y = self.fc5(x)

        return y