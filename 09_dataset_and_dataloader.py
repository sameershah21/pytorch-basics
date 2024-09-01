# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter
print("Chapter 9 DataSet and DataLoader")
# instead of using entire database for training data, we can use pytorch abilty to divide samples in batches.
#then we do optimization only on those batches therby saving lot of effort in training

#Terminologies: 
#epoch - it is one forward and backward pass of ALL training samples
#batch_size - number of training samples in one forward and backward pass
#number of iterations - number of passes where each pass uses [batch_size] number of samples
#Example: For 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class WineDataSet(Dataset):
    def __init__(self):
        #data loading
        xy=np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:,1:]) # slicing to get all rows  but start with col 1 and get everything till the end. Skip  column 1 is wine type
        self.y = torch.from_numpy(xy [:, [0]]) # slicing to get all rows but only col 0 as now we need the wine type here. i.e n_samples,1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index] # returns tuple
    
    def __len__(self):
        #len(dataset)
        return self.n_samples
    
#create dataset
dataset = WineDataSet()
# to check and understand the dataset lets do the following and then comment it out
# first_data = dataset[0]
# features,labels = first_data
# print (f"features:{features},\nlabels:{labels}") 
#features will show one row vector, labels with show label 1

# The dataset was loaded into the WineDataSet class, where the Wine column is used as the label (class) and all other columns are used as features.
# The first data point in the output corresponds to the first row in this table:
# Features: [14.23, 1.71, 2.43, 15.6, 127, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
# Label: [1]
batch_size=4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=16)

# dataiter=iter(dataloader)
# data = next(dataiter)
# features,labels = data
# print (f"features:{features},\nlabels:{labels}") 
# # since the batch size is 4, we will see 4 feature vectors. And for each feature vectors we will the corresponding (wine)class

#training loop
#define hyperparameters
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)
print(f"\ntotal_samples: {total_samples}, n_iterations: {n_iterations}")

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader): #unpack inputs and lables by enumerating. enum fn will give index and inputs,labels unpacked
        #forward and backward pass
        if (i+1) % 5 == 0:
            print(f"epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_iterations}, inputs: {inputs.shape}")
            # The dataset has 178 samples, Batch size is 4. hence each epoch has 178/45 = [44.5] = 45
            # inputs: torch.Size([4, 13] is because batch size is 4 and there are 13 features as shows above Features: [14.23, 1.71, 2.43, 15.6, 127, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
        #update weights
        
# if you dont want to create datasets, use can use torchvision's inbuilt dataset
# torchvision.datasets.MNIST()
# torchvision.datasets.FashionMNIST()
# torchvision.datasets.cifar()
# torchvision.datasets.coco()
