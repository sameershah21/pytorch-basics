# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter
print("Chapter 10 DataSet Transforms")
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class WineDataSet(Dataset):
    def __init__(self, transform=None):
        #data loading
        xy=np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        # self.x = torch.from_numpy(xy[:,1:]) # slicing to get all rows  but start with col 1 and get everything till the end. Skip  column 1 is wine type
        # self.y = torch.from_numpy(xy [:, [0]]) # slicing to get all rows but only col 0 as now we need the wine type here. i.e n_samples,1
        # note that we do not convert to tensor here
        self.x = xy[:, 1:]
        self.y = xy [:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index] # returns tuple
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        #len(dataset)
        return self.n_samples

class ToTensor:
    """
    Custom pytorch transformer that will convert numpy arrays to torch tensors
    """
    def __call__(self, sample):
        #unpack samples
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets) # return tuples

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
    

#create dataset
dataset1 = WineDataSet(transform=ToTensor())
dataset2 = WineDataSet(transform=None)



first_data = dataset1[0]
features,labels = first_data
print (f"features:{features},\nlabels:{labels}") 
print (f"feature type:{type(features)},\nlabel type:{type(labels)}") 
#features will show one row vector, labels with show label 1

print (f"------------") 
first_data = dataset2[0]
features,labels = first_data
print (f"features:{features},\nlabels:{labels}") 
print (f"feature type:{type(features)},\nlabel type:{type(labels)}") 

print (f"------------") 
#apply both transforms together by chaining them
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)]) # each feature value will be x4
dataset3 = WineDataSet(transform=composed)
#features will show one row vector, labels with show label 1
first_data = dataset3[0]
features,labels = first_data
print (f"features:{features},\nlabels:{labels}") 
print (f"feature type:{type(features)},\nlabel type:{type(labels)}") 