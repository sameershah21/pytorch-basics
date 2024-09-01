# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

# CNN typically work on image data. There are mainly 3 layers
# 1. Convolutional(Conv for short) layer
# 2. Activation Function (RELU)
# 3. Pooling layer - used to automatially learn some features from images
# 4. Fully Connected Layer (FC) - for actual classification tasks

# the conv filters work by applying a filter kernel to the image. Imagine a 1024x720 image matrix and filter size is 3x3. Filter is put on the first position (1x1) and output value 
# is calculated by multiplying and suming up all the values and then write the output value to output image at 1x1. Then we change the filter window position
# to move one position to the right. The first position of the filter was at the top-left corner (1,1) covering (1,1) to (3,3).
# With a stride of 1, the filter moves one pixel to the right. Top-left corner of the filter will move from (1,1) to (1,2). his means the filter now covers the region from (1,2) to (3,4).

# image may be of smaller size if filter does not fit the corner unless padding is used. Itrefers to the process of adding extra pixels around the edges of an input image before applying a convolutional filter

# Pooling is a downsampling operation commonly used in Convolutional Neural Networks (CNNs) to reduce the spatial dimensions (width and height) of the feature maps while retaining the most important information. The primary purpose of pooling is to reduce the computational complexity of the network, make the network invariant to small translations or distortions in the input image, and control overfitting by reducing the number of parameters in the model.
# Max pooling works by taking the maximum value within a pool size. For example: 
# 1  3  2  4
# 5  6  8  7
# 3  2  1  0
# 4  5  6  8  will give

#6  8  
#5  8 How?

# [1 3]     Max = 6
# [5 6]

# [2 4]     Max = 8
# [8 7]

# [3 2]     Max = 5
# [4 5]

# [1 0]     Max = 8
# [6 8]


import torch
import torch.nn as nn
import numpy as np
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plot
import torch.nn.functional as F
import time  # Import the time module

# Add this at the beginning of your script to start the timer
start_time = time.time()

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 0 prepare data
# hyper parameters
num_epochs = 4 # set to higher value and it will take more time
batch_size = 128 # use NVIDIA power!!! #4
learning_rate = 0.001

#dataset has PILImage of range [0,1]
# We transform them to Tensors of normalized range [1 -1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
#adding more transforms to increase accuracy
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])



# CIFAR
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform,download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform,download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

classes = ('plane', 'car', 'bird', 'cat', 
           'deer', 'deer', 'dog', 'frog', 'horse' , 'ship', 'truck')

# just for testing
# ---------------------------------
# def imgshow(img):
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plot.imshow(np.transpose(npimg, (1, 2, 0)))
# #get some random training images
# dataiter = iter(train_loader)
# images,labels = next(dataiter)
# #show images
# imgshow(torchvision.utils.make_grid(images))

# conv1 = nn.Conv2d(3, 6, 5)
# pool = nn.MaxPool2d(2, 2)
# conv2 = nn.Conv2d(6, 16, 5)
# print(images.shape)
# x = conv1(images)
# print(x.shape)
# x = pool(x)
# print(x.shape)
# x = conv2(x)
# print(x.shape)
# x = pool(x)
# print(x.shape)
# ---------------------------------

# Step 1 define model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__() # Initialize the parent class
        #origin shape is [4, 3, 32, 32] - batch size = 4, color channels = 3, 32x32=1024
        self.conv1 = nn.Conv2d(3, 6, 5) # input size, output size, kernel size
        #output size for conv layer is calculated by: (input width - filter size + (Padding * 2)) / Stride + 1
        # (32 - 5 + 1)/1 = 28 # remember image size is 32
        #conv1 shape is ([4, 6, 28, 28]) - batch size = 4, input size (equal to output size of previous layer) = 6, output size = 28 from above calc
        self.pool = nn.MaxPool2d(2, 2) #kernel size 2, stride is 2 (shifting window size)
        # 28/2 = 14 # pooling will reduce the size by factor of 2
        #pool shape is ([4, 6, 14, 14]) - batch size = 4, input size (equal to output size of previous conv layer) = 6, output size = 14 from above calc
        self.conv2 = nn.Conv2d(6, 16, 5) # input channel size = previous layer's output channel size
        #output size for conv layer is calculated by: (input width - filter size + (Padding * 2)) / Stride + 1
        # (14 - 5 + 1)/1 = 10  
        #conv2 shape is ([4, 16, 10, 10]) - batch size = 4, input size (equal to output size of previous conv layer) = 16, output size = 10 from above calc
        self.pool = nn.MaxPool2d(2, 2) #kernel size 2, stride is 2 (shifting window size)
        # 10/2 = 5 # pooling will reduce the size by factor of 2
        #pool2 shape is ([4, 16, 5, 5]) - batch size = 4, input size (equal to output size of previous conv layer) = 16, output size = 5 from above calc
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        # fully connected layers, we will have 3 of those (refer to video for architecture diagram)
        # also check the Linear layer will be exactly the size of previous pool layer so that we can continue, else it will error out here
        # output feature size of 120 wa randomly selected, it could be any value you want to use        
        self.fc2 = nn.Linear(120, 84) # input size (equals output size from previous layer), output size
        self.fc3 = nn.Linear(84, 10) # output size = 10 as we have 0-10 classes: cat, plane, bird, etc



    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # activation function wont change size. also need to be pooled         
        x = self.pool(F.relu(self.conv2(x))) # activation function wont change size. also need to be pooled        
        # before passsing to fully connected layer, it needs to be flatened 
        x = x.view(-1, 16*5*5) # -1, means automatcically calculate first size when other value is given, output size = 16 batches of 5x5 images
        x = F.relu(self.fc1(x)) # activation function wont change size
        x = F.relu(self.fc2(x)) # activation function wont change size
        x = self.fc3(x) # activation function wont change size. Dont use activation function on the final layer :) Remember not to use softmax as its included as part of CrossEntropyLoss()
        return x


model = ConvNet().to(device)

# Step 2 loss and optimzer
criterion = nn.CrossEntropyLoss() # applies softmax # when using multiclass classfication problem use cross entropy loss versus Binary Cross-Entropy Loss (BCELoss) or BCEWithLogitsLoss for Binary classification
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

# Step 3 training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader): # enum will give index (i) and data (tuple of images and labels). (remembers enum gives indes as well as corresponding value. useful when you want to keep track of data) 
        #origin shape is [4, 3, 32, 32] - batch size = 4, color channels = 3, 32x32=1024
        #input layer: 3 input channels ( 3 color channels (RGB) of the images),
        # 6 output channels -number of filters (or kernels) in the convolutional layer, 
        # which determines the number of output channels (feature maps) produced by this layer
        # 5 kernel size -The size of the convolutional filter is 5x5. This means the filter will slide over the image
        # with a window of 5x5 pixels, computing the convolution at each step.
        
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs,labels)

        #backward
        optimizer.zero_grad() # remember grad issue reset in computational as per previous chapters
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0: # every 100th step do the following
            print (f"epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss:.4f}")

print ("Finished Training")
# Calculate the total time taken in minutes
end_time = time.time()
total_time = (end_time - start_time) / 60  # Convert seconds to minutes
print(f"Total Training Time: {total_time:.2f} minutes")
# Step 4 testing and evaluation

with torch.no_grad(): # don't want to compute gradient for all the steps
    n_correct = 0 #  used to calculate the overall accuracy of the model.
    n_samples = 0 #  total number of samples in the test set. It is used as the denominator when calculating the overall accuracy of the model.
    n_class_correct = [0 for i in range(10)] #  This is a list (or array) where each element corresponds to a specific class. For example, n_class_correct[0] counts the number of correctly predicted samples for class 0 (e.g., 'plane'), n_class_correct[1] for class 1 (e.g., 'car'), and so on. This allows you to calculate the accuracy for each individual class.
    n_class_samples = [0 for i in range(10)] # Similar to n_class_correct, this is a list where each element corresponds to the total number of samples for a specific class. For example, n_class_samples[0] counts the total number of samples that belong to class 0 (e.g., 'plane'). This variable is used as the denominator when calculating the accuracy for each individual class.


    for images, labels in test_loader:        
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1) # will give outputs along with dimension which is 1 which is index. we don't care about values hence we put _ to ignore it
        n_samples += labels.size(0) # will give no. of samples in current batch (100)
        n_correct += (predictions == labels).sum().item() # for each correct prediction we add +1

        for i in range(labels.size(0)): # Using labels.size(0) instead of batch_size as the latter was giving error
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    
    acc = 100.0 * n_correct / n_samples

    print(f"accuracy of the network: {acc}%")

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"accuracy of the class name {classes[i]}: {acc}%")


