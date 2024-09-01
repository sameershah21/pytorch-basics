# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

import torch
import torch.nn as nn
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size) # layer 1
        self.relu = nn.ReLU() # activation function layer 2
        self.linear2 = nn.Linear(hidden_size, 1) # layer 3 output sizw will always be 1 since its a binary problem. for each possible class we have one output
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #since this uses pytorch BCE (Binary Cross Entropy) loss we need to implement sigmoid in the end to get the probablity
        y_pred = torch.sigmoid(out)
        return y_pred
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=5)
loss = nn.BCELoss() 
