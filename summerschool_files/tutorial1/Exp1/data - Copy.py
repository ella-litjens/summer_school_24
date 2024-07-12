import numpy as np      # to handle matrix and data operation
import pandas as pd     # to read csv and handle dataframe
import torch            # to load pytorch library
import torch.utils.data # to load data processor

# load training and testing datasets
train = pd.read_csv('mnist_train.csv', header=None)     # read a csv file called 'mnist_train.csv'
test = pd.read_csv('mnist_test.csv', header=None)       # read a csv file called 'mnist_train.csv'
print(train)
print(train.info())
print(test)
print(test.info())

# split image and label
train_label = train.iloc[:, 0].values      # first : gets all rows, second 0 gets the first column (label)
train_img = train.iloc[:, 1:]              # first : gets all rows, second 1: gets the second to last columns (data)
print(train_label)
print(train_img)
test_label = test.iloc[:, 0].values        # first : gets all rows, second 0 gets the first column (label)
test_img = test.iloc[:, 1:]                # first : gets all rows, second 1: gets the second to last columns (data)
print(test_label)
print(test_img)

print(train_img.shape)
print(test_img.shape)

# reshape data to be [samples][channel][width][height]
train_img = train_img.values.reshape(-1,1,28,28)
test_img = test_img.values.reshape(-1,1,28,28)
# convert image pixel value from [0, 255] to [0, 1]
train_img = train_img / 255.0
test_img = test_img / 255.0
print(train_img.shape)
print(test_img.shape)


# convert the data to tensor format for training
torch_X_train = torch.from_numpy(train_img).float()     # training images
torch_y_train = torch.from_numpy(train_label)           # training labels
torch_X_test = torch.from_numpy(test_img).float()       # testing images
torch_y_test = torch.from_numpy(test_label)             # testing labels
# print data dimension 
print('training image dimension: ', torch_X_train.shape)
print('training label dimension: ', torch_y_train.shape)
print('testing image dimension: ', torch_X_test.shape)
print('testing label dimension: ', torch_y_test.shape)

# pack image and label into one class
def get_train_set():
    return torch.utils.data.TensorDataset(torch_X_train,torch_y_train)

def get_test_set():
    return torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
