# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

# Transfer learning is a popular approach in deep learning where a model developed for one task is reused as the starting point 
# for a model on a different task. For example, you can train a model to classify birds and cats, and then modify only the last layer 
# of that model to classify bees and dogs. This approach allows for the rapid generation of new models because it avoids the need to 
# train an entirely new model from scratch, which can be very time-consuming, often taking days or even weeks.

# In transfer learning, we typically replace only the last layer of the pre-trained model and retrain this layer on the new task.
# Despite not retraining the entire model, transfer learning can still achieve excellent performance, making it a popular technique today.

# For instance, consider a Convolutional Neural Network (CNN) that has already been trained on a large dataset. 
# The model has optimized weights across all its layers. In transfer learning, we take this pre-trained model,
# modify the last fully connected layer, and train this layer on the new data. As a result, we create a new model that has been fine-tuned 
# for a specific task by adjusting only the final layer. This is the essence of transfer learning.

# ImageFolder
#scheduler
# Transfer Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plot
import time
import os
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# import data
data_dir = 'data/hymenoptera_data'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                    for x in ['train', 'val']} 

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
                for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print (f"class names: {class_names}")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print('-' * 10)

        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # backward + optimize only in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                #statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            #deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f"Training of complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model

#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.


# transfer model process begins
model = models.resnet18(pretrained=True) # pretained will give already optimized weights in imagenet data
#exchange last fully connected layer (by getting its input features)
num_features = model.fc.in_features
# create a new layer and assign it to last layer
model.fc = nn.Linear(num_features, 2) # 2 because we have 2 classes now
model.to(device)

# step 2 define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# step 3 (new): scheduler to update learning rate
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # every 7 epochs, multiply learning rate by 10%
# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train() # optimzer.step(9)
#     evaluate()
#     scheduler.step()  

model = train_model(model,criterion, optimizer, step_lr_scheduler, num_epochs=2)

# step 2 and step 3 is also called FINE TUNING MODEL. where we don't train the whole model again, but only fine tune all the weights based on new data and with new last layer.
print("-------------")

# another way to fine tune


model_conv = models.resnet18(pretrained=True) # pretained will give already optimized weights in imagenet data
#freeze all the layers in the beginning and only train last layer
for param in model.parameters():
    param.requires_grad = False


num_features = model.fc.in_features
# create a new layer and assign it to last layer
model_conv.fc = nn.Linear(num_features, 2) # 2 because we have 2 classes now
# this layer thought will have requires_grad = True by default
model_conv = model_conv.to(device)

# step 2 define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model.parameters(), lr=0.01)

# step 3 (new): scheduler to update learning rate
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1) # every 7 epochs, multiply learning rate by 10%
# for epoch in range(100):
#     train() # optimzer.step(9)
#     evaluate()
#     scheduler.step()  

model_conv = train_model(model_conv,criterion, optimizer_conv, exp_lr_scheduler, num_epochs=3)

#accuracy of model with transferred layer and fine tuning model will of course be low compared to
# main model
        
            
                                              


