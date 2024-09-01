# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

import torch
import torch.nn as nn
import numpy as np

# Cross-entropy loss is a metric used to evaluate the performance of a classification model, 
# where the model's output is a probability between 0 and 1. 
# It is particularly useful in multi-class problems. 
# The loss increases as the predicted probability diverges from the actual label,
# meaning that the closer the prediction is to the true label, the lower the loss will be.

# For example, consider two scenarios:
# In the first scenario, the prediction is accurate, resulting in a low cross-entropy loss.
# In the second scenario, the prediction is inaccurate, leading to a high cross-entropy loss.

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(f"softmax numpy: {outputs}") # softmax will squash the inputs and force the output to between 0 and 1. Output is a probablity, higher the value, higher the probablity

# calculate in pytorch
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(f"softmax pytorch: {outputs}") # softmax is probablity function, higher the value, higher the probablity. Sum of all values has to be 1.

# ---------

# One-hot encoding is used to represent categorical labels in a format suitable for machine learning models.
# For instance, if we have three possible classes—class 0, class 1, and class 2:
# class 0: [1 0 0]
# class 1: [0 1 0]
# class 2: [0 0 1]
# and the correct label is class 0, we represent it using one-hot encoding by placing a 1 in the position corresponding to class 0
# and 0s in the positions for the other classes. So, for this example ([2.0, 1.0, 0.1]), the one-hot encoded label would be [1, 0, 0].
# This method is applied to ensure that the model can effectively interpret categorical labels during training.

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float (predicted.shape[0]) - if we need to normalize it i,e divide by number of sample
# y must be one hot encoded:
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0 ,0])

# y_pred has probabilites
Y_pred_good = np.array([0.7, 0.2, 0.1]) # class 0 has high probablity
Y_pred_bad = np.array([0.1, 0.3, 0.6]) # class 0 has lowest probablity, class 2 gets high probablity
l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)

print(f"Loss 1 numpy: {l1:.4f}") # shows loss value (low is good)
print(f"Loss 2 numpy: {l2:.4f}") # shows loss value (high is bad)

# using torch

#When using pytorch cross-entropy loss in a model, it's important to note a couple of key points:

#Softmax Layer: Cross-entropy loss automatically applies the log-softmax function followed by the negative log likelihood loss, 
# so you should not manually implement a softmax layer in your model. Doing so would lead to incorrect results.

# Label Encoding: The target labels (Y) should not be one-hot encoded. Instead, you should provide the correct class label directly as a single integer.
# Similarly, the model's predictions should be provided as raw scores (logits) rather than probabilities obtained through a softmax function.

# These considerations are crucial to ensure that the cross-entropy loss function works correctly in your model.
loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
#nsamples x nclasses = 1 x 3
Y_pred_good = torch.tensor([[2.0 , 1.0,  0.1]])
Y_pred_bad = torch.tensor([[0.5 , 2.0,  0.3]])
                           
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f"Loss 1 torch: {l1:.4f}") # shows loss value (low is good)
print(f"Loss 2 torch: {l2:.4f}") # shows loss value (high is bad)

# get predictions
_, predictions1 = torch.max(Y_pred_good,1) #underscores are when you don't need that value
_, predictions2 = torch.max(Y_pred_bad,1)
print(f"Predictions1 torch: {predictions1}") # class 0 will be predicted, good prediction 
print(f"Predictions2 torch: {predictions2}") # class 1 will be predicted, bad prediction

print("----------")
# loss in pytorch allows for multiple samples. Lets check it out
# 3 samples
Y = torch.tensor([2,0,1])
#nsamples x nclasses = 3 x 3
Y_pred_good = torch.tensor([[0.1 , 1.0,  2.1],# since [2,0,1] class 2 must have highest value\ 
                            [2.0 , 1.0,  0.1],# since [2,0,1] class 0 must have highest value\
                            [1.0 , 2.0,  0.1]]) # since [2,0,1] class 1 must have highest value
Y_pred_bad = torch.tensor([[2.1 , 1.0,  0.1],# since [2,0,1] class 2 must have highest value\ 
                            [0.1 , 1.0,  2.1],# since [2,0,1] class 0 must have highest value\
                            [2.0 , 1.0,  3.1]]) # since [2,0,1] class 1 must have highest value
                           
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f"Loss 1 torch: {l1:.4f}") # shows loss value (low is good)
print(f"Loss 2 torch: {l2:.4f}") # shows loss value (high is bad)

# get predictions
_, predictions1 = torch.max(Y_pred_good,1) #underscores are when you don't need that value
_, predictions2 = torch.max(Y_pred_bad,1)
print(f"Predictions1 torch: {predictions1}") # class 0 will be predicted, good prediction 
print(f"Predictions2 torch: {predictions2}") # class 1 will be predicted, bad prediction