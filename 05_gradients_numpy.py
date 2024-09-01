# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

import numpy as np
print("Chapter 5 Gradient Descent with Autograd and Back propogation") 

# linear regression equation
# f = w * x
X = np.array([1,2,3,4], dtype=np.float32)

# Create the new array Y where each value is times 2 of X
Y = X * 2

w = 0.0

# model prediction
def forward(x):
    return w * x

# loss = MSE (Mean Square Error)
def loss(y, y_predicted): #y is actual value
    return ((y_predicted - y)**2).mean()

# gradient
# MSE = 1/N * ( w * x - y)**2. (prediction - actual value)squared/N
# derivative dJ/dw = 1/N 2*x (w*x -y)

def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean() # dot product = For two vectors 𝐴 and 𝐵 with A elements (A1,A2..An) and B elements (B1,B2...Bn). dot product is A.B = A1 * B1 + A2 * B2 + .... + An * Bn

print(f"Prediction before training: f(5) {forward(5):.3f}")

# Training 
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient
    dw = gradient(X,Y,y_pred)

    #update weights
    w -= learning_rate * dw #dw is gradient

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss={l:.8f}")

print(f"Prediction after training: f(5) {forward(5):.3f}")

print("\n now increase n_iters to 20 and epoch to 2 to get exact prediction")


