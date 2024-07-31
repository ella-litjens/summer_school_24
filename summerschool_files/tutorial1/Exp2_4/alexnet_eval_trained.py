import os
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torchvision.utils as utils
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import argparse

"""
input commands
"""
parser = argparse.ArgumentParser()
parser.add_argument("--test_img", type=str, default='whippet.jpg', help="testing image")
opt = parser.parse_args()

# function for visualizing the feature maps
def visualize_activation_maps(input, model):
    I = utils.make_grid(input, nrow=1, normalize=True, scale_each=True)
    img = I.permute((1, 2, 0)).cpu().numpy()

    conv_results = []
    x = input
    for idx, operation in enumerate(model.features):
        x = operation(x)
        if idx in {1, 4, 7, 9, 11}:
            conv_results.append(x)
    
    for i in range(5):
        conv_result = conv_results[i]
        N, C, H, W = conv_result.size()

        mean_acti_map = torch.mean(conv_result, 1, True)
        mean_acti_map = F.interpolate(mean_acti_map, size=[224,224], mode='bilinear', align_corners=False)

        map_grid = utils.make_grid(mean_acti_map, nrow=1, normalize=True, scale_each=True)
        map_grid = map_grid.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
        map_grid = cv2.applyColorMap(map_grid, cv2.COLORMAP_JET)
        map_grid = cv2.cvtColor(map_grid, cv2.COLOR_BGR2RGB)
        map_grid = np.float32(map_grid) / 255

        visual_acti_map = 0.6 * img + 0.4 * map_grid
        tensor_visual_acti_map = torch.from_numpy(visual_acti_map).permute(2, 0, 1)

        file_name_visual_acti_map = 'conv{}_activation_map.jpg'.format(i+1)
        utils.save_image(tensor_visual_acti_map, file_name_visual_acti_map)

    return 0

# main 
if __name__ == "__main__":
    """
    data transforms, for pre-processing the input testing image before feeding into the net
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),             # resize the input to 256x256
        transforms.CenterCrop(224),         # center crop the input to 224x224
        transforms.ToTensor(),              # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])

    # obtain the file path of the testing image
    test_image_dir = './alexnet_images'
    test_image_filepath = os.path.join(test_image_dir, opt.test_img)
    #print(test_image_filepath)

    # open the testing image
    img = Image.open(test_image_filepath)
    print("original image's shape: " + str(img.size))

    # Convert image to RGB if it has an alpha channel
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        raise ValueError("Image is not in RGB or RGBA format")
    print(f"Image mode after conversion (if applied): {img.mode}")

    # pre-process the input
    transformed_img = data_transforms(img)
    print("transformed image's shape: " + str(transformed_img.shape))
    # form a batch with only one image
    batch_img = torch.unsqueeze(transformed_img, 0)
    print("image batch's shape: " + str(batch_img.shape))

    # load pre-trained AlexNet model
    print("\nfeed the input into the pre-trained alexnet to get the output")
    
    #-------FOR PRE TRAINED MODEL --------
    #alexnet = models.alexnet(pretrained=True)
    #-------FOR OUR TRAINING-------
    alexnet = models.alexnet(pretrained=False)
    alexnet.classifier[6] = torch.nn.Linear(alexnet.classifier[6].in_features, 15)
    alexnet.load_state_dict(torch.load('/content/drive/MyDrive/CNN_tut/summer_school_24/summerschool_files/tutorial1/Exp2_4/checkpoints/AlexNet_epoch_30.pth'))
    #NOTE: the path above needs to be changed to the path to your last trained epoch
    
    # put the model to eval mode for testing
    alexnet.eval()

    # obtain the output of the model
    output = alexnet(batch_img)
    print("output vector's shape: " + str(output.shape))
    
    # obtain the activation maps
    visualize_activation_maps(batch_img, alexnet)

    # map the class no. to the corresponding label
    with open('class_names_15.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]
    
    # print the first 5 classes to see the labels
    #not sure whats up here i thought the point was to get it to 15 classes?
    print("\nprint the first 15 classes to see the lables")
    for i in range(15):
        print("class " + str(i) + ": " + str(classes[i]))
    
    # sort the probability vector in descending order
    sorted, indices = torch.sort(output, descending=True)
    percentage = F.softmax(output, dim=1)[0] * 100.0
    # obtain the first 5 classes (with the highest probability) the input belongs to
    results = [(classes[i], percentage[i].item()) for i in indices[0][:5]]
    print("\nprint the first 5 classes the testing image belongs to")
    for i in range(5):
        print('{}: {:.4f}%'.format(results[i][0], results[i][1]))
