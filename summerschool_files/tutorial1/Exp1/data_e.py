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

train_label = train.iloc[:, 0].values
train_img = train.iloc[:, 1:]
print(train_label)
print(train_img)

test_label = test.iloc[:, 0].values
test_img = test.iloc[:, 1:]
print(test_label)
print(test_img)

print(train_img.shape)
print(test_img.shape)

train_img = train_img.values.reshape(-1,1,28,28)
test_img = test_img.values.reshape(-1,1,28,28)
train_img = train_img / 255.0
test_img = test_img / 255.0
print(train_img.shape)
print(test_img.shape)
