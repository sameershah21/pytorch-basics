# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter
# we use Non-linear activation functions (which enable non-linear regression) instead of linear most of the time because:
# 1. If we only used linear functions (e.g., linear regression or layers without non-linear activation functions) in a neural network, the network would be equivalent to a single linear transformation, regardless of how many layers it has. This is because a composition of linear functions is still a linear function.
# 2. Introducing non-linear activation functions between layers allows the network to approximate a much wider variety of functions, including complex, non-linear relationships between the input and output. This makes neural networks highly flexible and capable of solving a broad range of problems.
# 3. how it works? Each layer in a neural network applies a linear transformation followed by a non-linear transformation. The non-linear activation function allows subsequent layers to interact in ways that create new feature representations. This process enables the network to build up layers of abstraction, learning increasingly complex features at each level.

import torch
import torch.nn as nn
import torch.nn.functional as F
# 2 ways to use activation functions
# option 1 create nn modules
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size) # layer 1
        self.relu = nn.ReLU() # activation function layer 2
        # other famous activation functions
        # nn.Sigmoid
        # nn.Softmax
        # nn.Tanh
        # nn.LeakyReLU
        self.linear2 = nn.Linear(hidden_size, num_classes) # layer 3 num_classes is output size. for each possible class we have one output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
   
        return out
    
# option 2 use activation functions directly in the forward pass
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size) # layer 1        
        self.linear2 = nn.Linear(hidden_size, num_classes) # layer 2  if binary  then (hidden_size, 1) as its just one layer
    
    def forward(self, x):        
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out)) 
        #NOTE sometimes unlike above where you use torch API to call the function torch.relu, etc,
        # some activation functions wont be able. In this case, you'd have to use torch.NN.functional
        #like F.relu
        
        # no softmax in the end since this uses pytorch cross entropy loss. If that wasn't the case, we would have to calculate softmax
        return out

model = NeuralNet(input_size=28*28, hidden_size=5, num_classes=3)
loss = nn.CrossEntropyLoss() # applies softmax    

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
loss = nn.CrossEntropyLoss() # applies softmax    
