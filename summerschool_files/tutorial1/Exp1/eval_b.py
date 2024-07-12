import numpy as np # to handle matrix and data operation
import torch # to load pytorch library
from PIL import Image
from network import LeNet5 # load network from network.py
import matplotlib.pyplot as plt
import argparse


#============================ parse the command line =============================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Train_test.pth", help="pre-trained model")
parser.add_argument("--image", type=str, help="image file")


opt = parser.parse_args()

#============================ start testing =============================================
# build the network
model = LeNet5('SGD')
if torch.cuda.is_available():
    model.cuda()
# load the pre-trained model
model_name = 'model/' + opt.model
if model_name:
	model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
	print('pretrained model is loaded')
# start testing mode
model.eval()

#======================== image processing =============================================
img_name = 'image/' + opt.image
print('load', img_name)
# read images
img = Image.open(img_name).convert('L')
# crop image as square shape
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
# resize image to dimension 28*28
img_crop = img_crop.resize((28,28), resample=Image.BICUBIC)
imarray = 1.0 - np.array(img_crop).reshape(1,1,28,28) / 255.0
# convert image as tensor format
if torch.cuda.is_available():
    data = torch.from_numpy(imarray).float().cuda()
else:
    data = torch.from_numpy(imarray).float()
output = model(data)
# obtain the prediction

# digits '0' to '9'; 
classes = ['digit ' + str(i) for i in range(10)]
# sort the output, and then pair up the classes and the percentage
# this command sorts the output in descending order
_, indices = torch.sort(output, descending=True)
# this command calculates the percentage of each class which the input belongs to
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
# this command pairs up the class label and the corresponding percentage
results = [(classes[i], percentage[i].item()) for i in indices[0][:]]
# print the probability of each class
for i in results:
    print(i)

with open('output.txt', 'w') as txt:
    for i in results:
        txt.write(str(i[0]) + ',' + str(i[1]) + '\n')
