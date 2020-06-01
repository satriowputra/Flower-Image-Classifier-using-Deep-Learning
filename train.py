# importing the packages
import pandas as pd
import numpy as np
# import matplotlib; matplotlib.use('agg')
# import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from time import time
from workspace_utils import active_session

# Dataset directory
data_dir = 'flowers'
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

# TODO: Build and train your network
# Make model select GPU as processing unit if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Selecting pre-trained model
model = models.densenet161(pretrained=True)
model

# Freze our parameter so there is no backprop through it
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(2208, 1000),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(1000,102),
                           nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Making sure we only train the classifier parameters and feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);

with active_session():
    start = time()
    epochs = 1
    steps = 0
    train_losses, validation_losses = [], []
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
            # Send model input and label tensor to selected device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        validation_loss += batch_loss.item()

                        # Calculating accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                now = time()
                train_losses.append(running_loss/len(trainloaders))
                validation_losses.append(validation_loss/len(validloaders))
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloaders):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloaders):.3f}.. "
                      f"Elapsed time: {(now-start)/3600:.2f}h")
                running_loss = 0
                model.train()
                
# plt.plot(train_losses, label='Training loss')
# plt.plot(validation_losses, label='Validation loss')
# plt.legend(frameon=False)
# plt.show();

# TODO: Do validation on the test set
start = time()
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
      
# TODO: Save the checkpoint
model.class_to_idx = train_datasets.class_to_idx
model.cpu()
state = {
          'tl_arch': 'densenet161',
          'epoch': epoch,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'class_to_idx': model.class_to_idx
        }
savepath='checkpoint.pth'
torch.save(state,savepath)