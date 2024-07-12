import torch 
import torch.nn as nn 
import torchvision 
import torchvision.models as models 

def AlexNet(num_classes):
    model = models.alexnet(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False 

    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)

    return model

    