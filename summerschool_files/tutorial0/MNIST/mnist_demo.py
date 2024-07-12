import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch # to load pytorch library
import torch.nn as nn # to load pytorch library
import torch.nn.functional as F # to load pytorch function
from PIL import Image
import torch.utils.data # to load data processor
from torch.autograd import Variable # pytorch data type
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Defining the network (LeNet-5)  
class LeNet5(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
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

    def forward(self, x):
        #Convolutional Layer/Pooling Layer/Activation
        x = self.conv1(x)
        x = self.act1(x)
        x = self.maxpool1(x)
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = self.conv2(x)
        x = self.act2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 400)
        #Fully Connected Layer/Activation
        x = self.fc1(x)
        x = self.act3(x)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        x = self.act4(x)

        return self.fc3(x)


model = LeNet5()
model.cuda()
model_name = 'model/Training_epoch_100.pth'
if model_name:
	model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
	print('pretrained model is loaded')
count=1
model.eval()
count = 1
for a in range (3):
    img_name = 'image/test' + str(a+1) + '.jpg'
    print('load', img_name)
    img = Image.open(img_name).convert('L')
    width, height = img.size
    if width > height:
      left = (width - height)/2
      top = 0
      right = width - (width - height)/2
      bottom = height
      img_crop = img.crop((left, top, right, bottom))
    else:
      left = 0
      top = (height - width)/2
      right = width
      bottom = height - (height - width)/2
      img_crop = img.crop((left, top, right, bottom))

    img_crop = img_crop.resize((28,28), resample=Image.BICUBIC)
    imarray = 1.0 - np.array(img_crop).reshape(1,1,28,28) / 255.0
    data = torch.from_numpy(imarray).float().cuda()
    output = model(data)
    _, predicted = torch.max(output.data, 1)
    print('prediction: ', predicted.item())
    # plt.subplot(1,3,count)
    # plt.subplots_adjust(hspace=0.2, wspace = 0.6)
    # count += 1
    # plt.axis('off')
    # plt.title('Prediction: ' + str(predicted.item()), fontsize=10)
    # plt.imshow(imarray[0,0,:,:], cmap='Greys', interpolation='None')