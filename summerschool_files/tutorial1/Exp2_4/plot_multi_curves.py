import os 
import numpy as np 
import pandas as pd 
import argparse
import matplotlib.pyplot as plt 

def get_arr_from_csv(csv_file):
    file_path = '{}.csv'.format(csv_file)
    df = pd.read_csv(file_path, header=None)
    np_array = df.to_numpy()

    train_loss = np_array[:,0]
    train_acc = np_array[:,1]
    test_loss = np_array[:,2]
    test_acc = np_array[:,3]

    return train_loss, train_acc, test_loss, test_acc


def plot_curves(data_list, file_list, model_name, curve_name, curve_type):
    i = 0
    for data in data_list:
        num_epochs = len(data)
        plt.plot(np.arange(1, num_epochs+1), data, label=file_list[i])
        if curve_type == 'Loss':
            plt.ylabel('Loss')
        else:
            plt.ylabel('Accuracy')
            plt.ylim(0.0, 1.0)
        plt.xlabel('Epoch no.')
        plt.grid(linestyle='--')
        plt.legend()
        i += 1

    fig_filename = '{}_{}_curve.png'.format(model_name, curve_name)
    plt.savefig(fig_filename, bbox_inches='tight')
    plt.clf()

    return True


"""
input commands
"""
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default='LeNet5', help="Model name. LeNet5 or AlexNet")

opt = parser.parse_args()

# main function
if __name__ == "__main__":
    # if opt.model_name == 'LeNet5':
    #     file_list = [
    #         'LeNet5_lr01_batchsize1000',
    #         'LeNet5_lr001_batchsize1000',
    #         'LeNet5_lr0001_batchsize1000',
    #         'LeNet5_lr01_batchsize100',
    #         'LeNet5_lr001_batchsize100',
    #         'LeNet5_lr0001_batchsize100',
    #         'LeNet5_lr01_batchsize10',
    #         'LeNet5_lr001_batchsize10',
    #         'LeNet5_lr0001_batchsize10'
    #     ]
    # elif opt.model_name == 'AlexNet':
    #     file_list = [
    #         'AlexNet_lr01_batchsize64',
    #         'AlexNet_lr001_batchsize64',
    #         'AlexNet_lr0001_batchsize64',
    #         'AlexNet_lr01_batchsize32',
    #         'AlexNet_lr001_batchsize32',
    #         'AlexNet_lr0001_batchsize32',
    #         'AlexNet_lr01_batchsize4',
    #         'AlexNet_lr001_batchsize4',
    #         'AlexNet_lr0001_batchsize4'
    #     ]
    if opt.model_name == 'LeNet5':
        file_list = [
            'LeNet5_lr01_batchsize1000',
            'LeNet5_lr001_batchsize1000'
        ]
    elif opt.model_name == 'AlexNet':
        file_list = [
            'AlexNet_lr001_batchsize64',
            'AlexNet_lr0001_batchsize64'
        ]

    train_loss = []
    train_acc = [] 
    test_loss = [] 
    test_acc = []

    for csv_file in file_list:
        tmp_train_loss, tmp_train_acc, tmp_test_loss, tmp_test_acc = get_arr_from_csv(csv_file)
        train_loss.append(tmp_train_loss)
        train_acc.append(tmp_train_acc)
        test_loss.append(tmp_test_loss)
        test_acc.append(tmp_test_acc)

    plot_curves(train_loss, file_list, opt.model_name, 'train_loss', 'Loss')
    plot_curves(train_acc, file_list, opt.model_name, 'train_acc', 'Accuracy')
    plot_curves(test_loss, file_list, opt.model_name, 'test_loss', 'Loss')
    plot_curves(test_acc, file_list, opt.model_name, 'test_acc', 'Accuracy')

