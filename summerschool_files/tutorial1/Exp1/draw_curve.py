import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import matplotlib.pyplot as plt

#=========================== load txt files =======================
train_loss = np.loadtxt('train_loss_record.txt')
train_accuracy = np.loadtxt('train_accuracy_record.txt')
test_accuracy = np.loadtxt('test_accuracy_record.txt')
#=========================== plot the training and testing curves =======================
# generate a figure for plot
fig = plt.figure()
# plot training loss
plt.plot(np.arange(100), train_loss, color='blue')
# plot training accuracy
plt.plot(np.arange(100), train_accuracy, color='green')
# plot testing accuracy
plt.plot(np.arange(100), test_accuracy, color='red')
# output legend
plt.legend(['Train Loss', 'Train accuracy', 'Test accuracy'], loc='upper right')
# define x- and y- labels
plt.xlabel('number of epoches')
plt.ylabel('loss')
# show figures
fig.savefig('train_curve.png')