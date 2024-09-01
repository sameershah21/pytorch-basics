# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter
import torch
print("Chapter 3 autogradient") 

# Gradient Descent is a method to minimize the loss function
# (a measure of how far off the model's predictions are from the actual results) 
# by iteratively moving towards the minimum point in the loss landscape.
# In a typical Gradient Descent, the gradients of the loss function 
# with respect to the model's parameters are calculated for the entire dataset,
# which is then used to update the model parameters in the opposite direction
# of the gradient to reduce the loss.

x= torch.randn(3, requires_grad=True)
print("\nNormal Distribution (Gaussian): torch.randn(3) generates a tensor with random numbers \
drawn from a normal distribution (also known as a Gaussian distribution) \
with a mean of 0 and a standard deviation of 1. This means that the numbers are centered around 0 \
and can be both positive and negative, with most of them being close to 0. \
Useful when you need random values that follow a normal distribution, \
often used in statistical modeling or neural network weight initialization")
print("\n when requires_grad=True pytorch tracks that it requires a gradient. now when we do operations with tensors, pytorch will create computation graph.")
print(x)

y = x + 2
print (y)

z = y * y * 2
print(z)

z=z.mean()
print(z)

print(f"\n to calculate gradient (derivative dz/dx):{z.backward()}. This will create a vector Jacobian product (ask chatgpt to explain it to 10 year old, basically, \
      its understandng what amount of each ingredient you have to take to make curry, dal and other similar products. things like rai, jeeru, hing).")
print(x.grad) #x.grad stores gradients
print("-----------")
print ("\n check what happens if we don't specify requires_grad=True")

x= torch.randn(3)
print(x)

y = x + 2
print (y)

z = y * y * 2
print(z)

z=z.mean()
print(z)

# print(f"\n to calculate gradient (derivative dz/dx):{z.backward()}") #will cause error
# print(x.grad) #x.grad stores gradients

print("-----------")

print("\n What if z was not a scalar value? in above example i.e it had direction in addition to amount")
x= torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print (y)

z = y * y * 2
print(z)

#z=z.mean() # did this to make it a non scalar value
print(z)
# z.backward() this will error -  grad can be implicitly created only for scalar outputs
print ("\n to fix this, we have to give it a gradient argument by creating a vector of same size (3)")
v = torch.tensor([0.1,1.0,0.001],dtype=torch.float32)
z.backward(v) # add vector to backward function
print(x.grad)

print("-----------")

print("\n How to prevent tracking the gradient so that gradient dosen't track backpropogation history in computation graph \
      It is done using 3 ways")
x1=x2=x3= torch.randn(3, requires_grad=True) 
print(x)
# option 1
x1.requires_grad_(False)  #remember that trailing underscore is in place modification  i.e it will end up modifying the x
print(f" Using requires_grad_:{x1}")
# option 2 - create a new tensor that won't require gradient
y=x2.detach
print(f" Using detach which will create a new tensor that won't require gradient:{x2}")
# option 3 - using no_grad
with torch.no_grad():
    y = x3+2
    print(f" Using torch.no_grad:{y}")

print("-----------")

print("Whenever a backward function is called, then the gradient for the tensor will accumulated in the .grad attribute. So the values will be summed up")

weights = torch.ones(4, requires_grad=True) 
print("\nwhen we loop once")
for epoch in range(1):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)
print("\nhowever, if we call loop twice")
for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)
print("\nhowever, if we call loop 3x")
for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

print("\nhence we need to make them 0")
for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()

print("\n ----------- torch optimizer -------")
weights = torch.ones(4, requires_grad=True)
optimizer= torch.optim.SGD([weights], lr=0.01) # lr is learning rate
optimizer.step()
optimizer.zero_grad()