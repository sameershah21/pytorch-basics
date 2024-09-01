# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter


# A general training pipeline in pytorch consits of there steps
# 1 - Design model (input size, output size, forward pass)
# 2 - Construct loss and optimizer
# 3 - Training loop
        # - forward pass: compute prediction
        # - backward pass: gradients
        # - update weights
import torch
import torch.nn as nn #neural nw has loss and optimizer replacements instead of us doing it manually as in previous chapter 
import numpy as np
from sklearn import datasets # to generate a regression dataset
import matplotlib.pyplot as plt # to plot 
print("Chapter 7 Linear Regression")

#Step 0 prepare data
X_numpy,Y_numpy = datasets.make_regression(n_samples=100, n_features =1, \
                                           noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

Y = Y.view(Y.shape[0],1) #reshape Y as data only has one row and we want to make it a column vector.
#so we want to put each value in 1 row and whole shape has only one column


# linear regression equation
# f = w * x
#X = torch.tensor([1,2,3,4], dtype=torch.float32)
#X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) # X changes a bit to fit nn. Linear model input


# Create the new array Y where each value is times 2 of X
#Y = X * 2

n_samples, n_features = X.shape
print(f"n_samples: {n_samples}, n_features: {n_features}")
#1. define model

input_size = n_features
output_size = n_features
# w = torch.tensor(0.0,dtype=torch.float32, requires_grad=True)

# # model prediction
# def forward(x):
#     return w * x

# Above code for weights and model creation is commented as now we use nn's inbuilt model
# model = nn.Linear(input_size,output_size)
#if we had to create our own model, we would do it something like to foll:
class LinearRegression(nn.Module):
    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    # above code shows what init really has in it
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression,self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)
    
model = LinearRegression(input_size,output_size)

# loss = MSE (Mean Square Error)
# def loss(y, y_predicted): #y is actual value
#     return ((y_predicted - y)**2).mean()

# gradient
# MSE = 1/N * ( w * x - y)**2. (prediction - actual value)squared/N
# derivative dJ/dw = 1/N 2*x (w*x -y)

# def gradient(x,y,y_predicted):
#     return np.dot(2*x,y_predicted-y).mean() # dot product = For two vectors 𝐴 and 𝐵 with A elements (A1,A2..An) and B elements (B1,B2...Bn). dot product is A.B = A1 * B1 + A2 * B2 + .... + An * Bn

# print(f"Prediction before training: f(5) {forward(5):.3f}")
X_test = torch.tensor([5],dtype=torch.float32)
print(f"Prediction before training: f(5) {model(X_test).item():.3f}")

# 2. Define loss and optimizer 
learning_rate = 0.01
loss = nn.MSELoss()
# optimizer = torch.optim.SGD([w], lr=learning_rate)  #commented since now we don't use weights and instead use nn mode
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Gradient Descent is a method to minimize the loss function
# (a measure of how far off the model's predictions are from the actual results) 
# by iteratively moving towards the minimum point in the loss landscape.
# In a typical Gradient Descent, the gradients of the loss function 
# with respect to the model's parameters are calculated for the entire dataset,
# which is then used to update the model parameters in the opposite direction
# of the gradient to reduce the loss.

# SGD stands for Stochastic Gradient Descent.Stochastic means random. 
# In the context of SGD, it refers to the idea of using a random subset
#  (or even a single data point) of the training data to
#  calculate the gradient and update the model parameters, rather than using the entire dataset
#  as in traditional Gradient Descent.
# 3. Training

n_iters = 100
for epoch in range(n_iters):
    # prediction = forward pass
    # y_pred = forward(X) # commented out since now we will use nn linear model
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backward pass

    #dw = gradient(X,Y,y_pred)
    l.backward() #dl/dw

    #update weights

    # with torch.no_grad():
    #     w -= learning_rate * w.grad#dw
    #     #dw is gradient
    #     #NOTE: with torch.no_grad: as we don't want this to part of computational graph
    #     # i.e we dont want to track it (since w has requires_grad=True)
    optimizer.step() # no need to manually update weights as nn as function to do this, hence commented out above code

    # print("Whenever a backward function is called, then the gradient for the tensor will accumulated \
    #     in the .grad attribute. So the values will be summed up \
    #     Hence, we empty the gradients")

    # w.grad.zero_()
    optimizer.zero_grad() 

    

    if epoch % 10 == 0:
        #need to unpack w from the nn linear output
        [w,b] = model.parameters() #b = bias and its optional
        # print(f"epoch {epoch+1}: w = {w:.3f}, loss={l:.8f}")
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss={l:.8f}")

# print(f"Prediction after training: f(5) {forward(5):.3f}")
print(f"Prediction after training: f(5) {model(X_test).item():.3f}")

#plot
predicted = model(X).detach() # before calling the model and converting to numpy,
# we need to detach the model to prvenet it from coming back in computation graph. Remember the requires_grad=True issue mentioned in previous chapters?
# Hence, the above will generate a new tensor rather then modifying the existing tensor
predicted = predicted.numpy()
plt.plot(X_numpy, Y_numpy, 'ro') # ro is Red circles
plt.plot(X_numpy, predicted, 'b') # blue
plt.show()


print("\n now increase n_iters to 100 and epoch to 10 to get exact prediction. \
      NOTE we had to increase these values from numpy because back propagation is not as \
      exact as numerical gradient computation")
#Update for chapter 6, running this again and again will (randomly) get better prediction for example, first time it will be 9.444 second time it will be 9.594
# this is because of SGD


