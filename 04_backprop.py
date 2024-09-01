# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

import torch
print("Chapter 4 Back propogation") 
x = torch.tensor(1.0)
y=torch.tensor(2.0)
w=torch.tensor(1.0, requires_grad=True)
# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2 #(sqaured)

print (loss)

# backward pass
loss.backward()
print(w.grad)

# update the weights
# next forward and backward pass

# continue optimizing:
# update weights, this operation should not be part of the computational graph
with torch.no_grad():
    w -= 0.01 * w.grad
# don't forget to zero the gradients
w.grad.zero_()

# next forward and backward pass...