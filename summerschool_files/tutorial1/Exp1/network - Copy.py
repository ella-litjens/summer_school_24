import torch # to load pytorch library
import torch.nn as nn # to load pytorch library

# Defining the network (LeNet-5)  
class LeNet5(nn.Module):
    # This defines the structure of the NN.
    def __init__(self, initialization):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    # Weights and biases initialization
        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                if initialization == 'Kaiming':
                    torch.nn.init.kaiming_normal_(m.weight)
                elif initialization == 'Xavier':
                    torch.nn.init.kaiming_normal_(m.weight)
                elif initialization == 'Random':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('Linear') != -1:
                if initialization == 'Kaiming':
                    torch.nn.init.kaiming_normal_(m.weight)
                elif initialization == 'Xavier':
                    torch.nn.init.kaiming_normal_(m.weight)
                elif initialization == 'Random':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Convolutional Layer, Activation and MaxPooling Layer
        x = self.conv1(x) 
        x = self.act1(x)
        x = self.maxpool1(x)
        # Convolutional Layer, Activation and MaxPooling Layer
        x = self.conv2(x)
        x = self.act2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 400)
        # Fully Connected Layer and Activation
        x = self.fc1(x)
        x = self.act3(x)
        # Fully Connected Layer and Activation
        x = self.fc2(x)
        x = self.act4(x)
        # Convolutional Layer, Activation and MaxPooling Layer
        output = self.fc3(x)

        return output