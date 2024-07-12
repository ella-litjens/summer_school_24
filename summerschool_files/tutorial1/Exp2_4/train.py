import numpy as np 
import os
import torch
import time
import math
import pandas as pd 
import matplotlib.pyplot as plt 

def plot_train_curve(train_info, model_name, learning_rate, batch_size):
    if learning_rate == 0.1:
        lr = '01'
    elif learning_rate == 0.01:
        lr = '001'
    elif learning_rate == 0.001:
        lr = '0001'
    else:
        lr = 'Selfdefined'

    train_info = np.array(train_info)
    train_loss = train_info[:,0]
    train_acc = train_info[:,1]
    test_loss = train_info[:,2]
    test_acc = train_info[:,3]
    num_epochs = len(train_loss)

    plt.plot(np.arange(1, num_epochs+1), train_loss)
    plt.plot(np.arange(1, num_epochs+1), test_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch no.')
    plt.grid(linestyle='--')
    plt.legend(['Train Loss', 'Test Loss'])
    fig_filename = '{}_lr{}_batchsize{}_loss_curve.png'.format(model_name, lr, batch_size)
    plt.savefig(fig_filename, bbox_inches='tight')
    plt.clf()

    plt.plot(np.arange(1, num_epochs+1), train_acc)
    plt.plot(np.arange(1, num_epochs+1), test_acc)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch no.')
    plt.ylim(0.0, 1.0)
    plt.grid(linestyle='--')
    plt.legend(['Train Acc', 'Test Acc'])
    fig_filename = '{}_lr{}_batchsize{}_acc_curve.png'.format(model_name, lr, batch_size)
    plt.savefig(fig_filename, bbox_inches='tight')
    plt.clf()

    return True


def write_train_info(train_info, model_name, learning_rate, batch_size):
    if learning_rate == 0.1:
        lr = '01'
    elif learning_rate == 0.01:
        lr = '001'
    elif learning_rate == 0.001:
        lr = '0001'
    else:
        lr = 'Selfdefined'

    filename = '{}_lr{}_batchsize{}.csv'.format(model_name, lr, batch_size)
    df = pd.DataFrame(train_info)
    df.to_csv(filename, header=False, index=False)

    return True


def save_checkpoints(model, save_path, model_name, epoch):
    filename = '{}_epoch_{}.pth'.format(model_name, epoch)
    torch.save(model.state_dict(), os.path.join(save_path, filename))

    return True


def train_and_test(opt, model, criterion, optimizer, train_data_loader, test_data_loader, dataset_sizes, device):
    since = time.time()

    num_epochs = opt.epochs
    num_train_iter = math.ceil(dataset_sizes['train'] / opt.batch_size)
    num_test_iter = math.ceil(dataset_sizes['test'] / opt.batch_size)

    train_info = []

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            # forward pass
            y = model(inputs)
            _, preds = torch.max(y.data, 1)
            loss = criterion(y, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            running_corrects = preds.eq(labels.data.view_as(preds))

            acc = torch.mean(running_corrects.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        # testing
        with torch.no_grad():
            model.eval() 

            for batch_idx, (inputs, labels) in enumerate(test_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass
                y = model(inputs)
                _, preds = torch.max(y.data, 1)
                loss = criterion(y, labels)

                test_loss += loss.item() * inputs.size(0)
                running_corrects = preds.eq(labels.data.view_as(preds))

                acc = torch.mean(running_corrects.type(torch.FloatTensor))

                test_acc += acc.item() * inputs.size(0)

        # complete one epoch
        epoch_train_loss = train_loss / dataset_sizes['train']
        epoch_train_acc = train_acc / dataset_sizes['train']

        epoch_test_loss = test_loss / dataset_sizes['test']
        epoch_test_acc = test_acc / dataset_sizes['test']

        print('Epoch {}/{}, Train loss: {:.4f}, Train acc: {:.4f}'.format(epoch+1, num_epochs, epoch_train_loss, epoch_train_acc))
        print('Epoch {}/{}, Test loss: {:.4f}, Test acc: {:.4f}'.format(epoch+1, num_epochs, epoch_test_loss, epoch_test_acc))
        train_info.append([epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc])

        save_checkpoints(model, opt.save_path, opt.model_name, epoch+1)

    time_elapsed = time.time() - since 
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    write_train_info(train_info, opt.model_name, opt.lr, opt.batch_size)
    plot_train_curve(train_info, opt.model_name, opt.lr, opt.batch_size)

    return model, train_info
