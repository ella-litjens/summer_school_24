import argparse
import numpy as np 
import os
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as transforms 

from data import get_train_and_test_data
from lenet import LeNet5
from train import train_and_test

"""
input commands
"""
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

parser.add_argument("--data_path", type=str, default='datasets/MNIST', help="Path to dataset")
parser.add_argument("--save_path", type=str, default='checkpoints', help="Path to save your checkpoints")
parser.add_argument("--model_name", type=str, default='LeNet5', help="Model name. LeNet5 or AlexNet")

parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="SGD weight decay (default: 5e-4)")

opt = parser.parse_args()

# main function
if __name__ == "__main__":
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    }
    
    train_data_loader, test_data_loader, dataset_sizes, class_names, num_classes = get_train_and_test_data(opt.data_path, data_transforms, opt.batch_size)

    print("\nDefine model and loss function ...")
    model = LeNet5(num_classes)
    criterion = nn.CrossEntropyLoss()
    print("Completed")

    print("\nMoving to GPU if possible ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    print(device)
    print("Completed")

    print("\nSetting optimizer ...")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    print("Setting optimizer completed")

    print("\nStart training ...")
    model = train_and_test(opt, model, criterion, optimizer, train_data_loader, test_data_loader, dataset_sizes, device)
    print("Training completed")

    