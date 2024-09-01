# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

import torch
print("Chapter 5 Gradient Descent with Autograd and Back propogation") 

# linear regression equation
# f = w * x
X = torch.tensor([1,2,3,4], dtype=torch.float32)

# Create the new array Y where each value is times 2 of X
Y = X * 2

w = torch.tensor(0.0,dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss = MSE (Mean Square Error)
def loss(y, y_predicted): #y is actual value
    return ((y_predicted - y)**2).mean()

# gradient
# MSE = 1/N * ( w * x - y)**2. (prediction - actual value)squared/N
# derivative dJ/dw = 1/N 2*x (w*x -y)

# def gradient(x,y,y_predicted):
#     return np.dot(2*x,y_predicted-y).mean() # dot product = For two vectors 𝐴 and 𝐵 with A elements (A1,A2..An) and B elements (B1,B2...Bn). dot product is A.B = A1 * B1 + A2 * B2 + .... + An * Bn

print(f"Prediction before training: f(5) {forward(5):.3f}")

# Training 
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backward pass

    #dw = gradient(X,Y,y_pred)
    l.backward() #dl/dw

    #update weights

    with torch.no_grad():
        w -= learning_rate * w.grad#dw
        #dw is gradient
        #NOTE: with torch.no_grad: as we don't want this to part of computational graph
        # i.e we dont want to track it (since w has requires_grad=True)
    
    print("Whenever a backward function is called, then the gradient for the tensor will accumulated \
        in the .grad attribute. So the values will be summed up \
        Hence, we empty the gradients")
    w.grad.zero_()
    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss={l:.8f}")

print(f"Prediction after training: f(5) {forward(5):.3f}")

print("\n now increase n_iters to 100 and epoch to 10 to get exact prediction. \
      NOTE we had to increase these values from numpy because back propagation is not as \
      exact as numerical gradient computation")


