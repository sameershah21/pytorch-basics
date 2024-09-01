# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter
# Read https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
#Lambda LR https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR
# 1. Introduction to Learning Rate Schedulers
# Purpose: The tutorial introduces the concept of a learning rate scheduler, which dynamically adjusts the learning rate during the training process.

# Why It's Important: The learning rate is one of the most critical hyperparameters in training neural networks. Properly adjusting it can lead to faster convergence and better model performance.

# 2. Types of Learning Rate Adjustments
# Based on Epochs: Adjusting the learning rate after a certain number of training epochs.

# Based on Validation Metrics: Modifying the learning rate when specific validation metrics (like loss or accuracy) stop improving.

# 3. PyTorch's Learning Rate Scheduler API
# Availability: PyTorch provides several built-in methods within the torch.optim.lr_scheduler module to facilitate learning rate adjustments.

# Ease of Implementation: Integrating these schedulers into your training loop is straightforward, requiring only a few additional lines of code.

# 4. Best Practices
# Decreasing vs. Increasing Learning Rate: Typically, the learning rate is decreased over time to allow the model to fine-tune its weights. However, the specific adjustment strategy should align with the problem at hand.

# Scheduler Placement: It's crucial to apply the learning rate scheduler after the optimizer's update step. This ensures that the learning rate adjustments take effect in the next iteration.

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10, 1)

optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

# 1. Lambda LR
lambda1 = lambda epoch: epoch / 10 # lambda function divide epoch by 10 and mutliply by lr. 
# for example : for 1st epoch 1/10 * 0.1, for 2nd epoch 2/10 * 0.1, .... 
# Note that in this example the lr is increasing but usually it should be decreasing. we will see that below in next section
scheduler = lr_scheduler.LambdaLR(optimizer, lambda1)

print(f"\nLambda LR Optimzer state_dict: {optimizer.state_dict()}")
for epoch in range(5):
    # loss.backward()
    optimizer.step()
    # validate(...)
    scheduler.step()
    print(f"{optimizer.state_dict()['param_groups'][0]['lr']}")

#-----------
# 2. Multplicative LR
lr = 0.1
model = nn.Linear(10, 1)
optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
lambda2 = lambda epoch: 0.95
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda2) # here we can check tht lr will decrease over epochs
print(f"\nMultiplicative  LR Optimzer state_dict: {optimizer.state_dict()}")
for epoch in range(5):
    # loss.backward()
    optimizer.step()
    # validate(...)
    scheduler.step()
    print(f"{optimizer.state_dict()['param_groups'][0]['lr']}")

#----------
# Step LR very powerful and simple https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

#----------
# MultiStep LR (can use different step sizes per sets of epochs) https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR

#----------
# Exponential LR (same as step LR but with step size always equals 1)
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR

#----------
# Reduce LR OnPlateau ( not dependent on epoch, but uses other other measurement metrics. it  reduces the lr when the metric stops improving)
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau

# check same docs for more