# importing the packages
import argparse
import numpy as np
import json

import torch
from torch import nn
from torchvision import transforms, models

# Defining arguments or command line option
parser = argparse.ArgumentParser(description='Arguments',)

parser.add_argument('image_path', action='store')
parser.add_argument('checkpoint_dir', action='store')
parser.add_argument('--top_k', action='store', default=5, dest='top_k',
                    type=int)
parser.add_argument('--category_names', action='store', 
                    default='cat_to_name.json', dest='category_names')
parser.add_argument('--gpu', action='store_true', default=False, 
                    dest='boolean_gpu')

results = parser.parse_args()

# Open category names
with open(results.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Select processing unit as per user input
if results.boolean_gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Predicting will be using CUDA as specified")
    else:
        device = torch.device("cpu")
        print("Cuda device is not availabe. Predicting will continue using CPU")
else:
    device = torch.device("cpu")
    print("Predicting will be using CPU")
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
if device.type == 'cpu':
    checkpoint = torch.load(results.checkpoint_dir, map_location='cpu')
else:
    checkpoint = torch.load(results.checkpoint_dir, map_location='cuda:0')

if checkpoint['tl_arch'] == 'densenet161':
    model = models.densenet161(pretrained=True)

elif checkpoint['tl_arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)

else:
    print("TL architecture is not supported")
    
for param in model.parameters():
    param.required_grad = False

model.classifier = checkpoint['classifier']
model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['state_dict'])
# model.to(device)
model.eval()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    from PIL import Image
    input_image = Image.open(image)
    
    # Resizing image
    width, height  = input_image.size
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    elif width > height:
        new_height = 256
        new_width = int(aspect_ratio*256)
    else:
        new_width = 256
        new_height = 256
    input_image = input_image.resize((new_width, new_height))

    # Cropping image
    width, height  = input_image.size
    left = (width-224)/2
    right = left + 224
    lower = (height-224)/2
    upper = lower + 224

    input_image = input_image.crop((left, lower, right, upper))

    np_image = np.array(input_image) / 255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image - means) / stds

    norm_image = norm_image.transpose((2, 0, 1))

    return norm_image


def predict(image_path, model, topk=results.top_k):
    ''' 
    Predict the class (or classes) of an image using a 
    trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    img = process_image(image_path)
    if device.type == 'cuda':
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    else:
        img = torch.from_numpy(img).type(torch.FloatTensor)
    img.to(device)
    img_input = img.unsqueeze(0)
    logps = model.forward(img_input)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    top_p = top_p.cpu().detach().numpy().tolist()[0]
    top_class = top_class.cpu().detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_class = [idx_to_class[label] for label in top_class]
    return top_p, top_class

# TODO: Display an image along with the top 5 classes
img_prc = process_image(results.image_path)

probs, classes = predict(results.image_path, model)

idx_to_class = {val: key for key, val in model.class_to_idx.items()}

classes_cat = [cat_to_name[label] for label in classes]

print(classes_cat)
print(probs)
