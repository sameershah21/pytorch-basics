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
        self.linear2 = nn.Linear(hidden_size, num_classes) # layer 3 num_classes is output size. for each possible class we have one output
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax in the end since this uses pytorch cross entropy loss. If that wasn't the case, we would have to calculate softmax
        return out
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
loss = nn.CrossEntropyLoss() # applies softmax    
