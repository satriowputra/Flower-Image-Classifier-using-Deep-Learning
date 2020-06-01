# importing the packages
import argparse
import pandas as pd
import numpy as np
import json
# import matplotlib; matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import seaborn as sb

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from time import time
from workspace_utils import active_session

parser = argparse.ArgumentParser(description='Arguments',)

parser.add_argument('data_directory', action='store')
parser.add_argument('checkpoint_dir', action='store')
parser.add_argument('--top_k', action='store', default=5, dest='top_k', type=int)
parser.add_argument('--category_names', action='store', default='cat_to_name.json', dest='category_names')
parser.add_argument('--gpu', action='store_true', default=False, dest='boolean_gpu')

results = parser.parse_args()

# Dataset directory
data_dir = results.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64)

with open(results.category_names, 'r') as f:
    cat_to_name = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Write a function that loads a checkpoint and rebuilds the model
if results.boolean_gpu == False:
    checkpoint = torch.load(results.checkpoint_dir, map_location='cpu')
elif results.boolean_gpu == True:
    checkpoint = torch.load(results.checkpoint_dir)

if checkpoint['tl_arch'] == 'densenet161':
    model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.required_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(nn.Linear(2208, 1000),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(1000,102),
                              nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

else:
    print("TL architecture is not correct")
    
# Testing loaded network
start = time()
model.to(device);
running_loss = 0
validation_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)

        # Calculating accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

now = time()

print(f"Test accuracy: {accuracy/len(testloaders):.3f}.. "
      f"Elapsed time: {(now-start)/3600:.2f}h")

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    from PIL import Image
    input_image = Image.open(image)
    input_image = input_image.resize((256, 256))

    width = input_image.width
    height = input_image.height

#     (left, upper, right, lower) = (((width-224)/2), 224, 224, ((height-224)/2))
    left = (width-224)/2
    upper = left + 224
    right = left + 224
    lower = (height-224)/2

    input_image = input_image.crop((left, lower, right, upper))

    np_image = np.array(input_image) / 255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image - means) / stds

    norm_image = norm_image.transpose((2, 0, 1))

    return norm_image

# def imshow(image, ax=None, title=None):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()
    
#     if title:
#         plt.title(title)

#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.transpose((1, 2, 0))
    
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
    
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
    
#     ax.imshow(image);
    
#     return ax

# image = 'flowers/test/30/image_03466.jpg'

# input = process_image(image)

# imshow(input)

def predict(image_path, model, topk=results.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.cpu()
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img_input = img.unsqueeze(0)
    logps = model.forward(img_input)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    top_p = top_p.detach().numpy().tolist()[0]
    top_class = top_class.detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_class = [idx_to_class[label] for label in top_class]
    return top_p, top_class

# TODO: Display an image along with the top 5 classes
image = 'flowers/test/30/image_03466.jpg'

img_prc = process_image(image)

# plt.figure(figsize=(10,8))
# ax = plt.subplot(2, 1, 1)
# title = cat_to_name[image.split('/')[2]]
# imshow(img_prc, ax=ax, title=title)

probs, classes = predict(image, model)

idx_to_class = {val: key for key, val in model.class_to_idx.items()}

classes_cat = [cat_to_name[label] for label in classes]
# plt.subplots(figsize=(7,8))
# plt.subplot(2, 1, 2)
# color = sb.color_palette()[0]
# sb.barplot(y=classes_cat, x=probs, color=color);

print(classes_cat)
print(probs)


