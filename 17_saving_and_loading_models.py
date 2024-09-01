# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

import torch
import torch.nn as nn

# Only following 3 functions are used to save and load
# torch.save(arg, PATH) # can save an tensors, models or any dict as param for saving. Uses python's pkl module to serialize the object (non human readable) and save them. 
# torch.load(PATH)
# model.load_state_dict(arg)  

#there are 2 ways to save and load the model
# 1. Lazy method: Saves complete model. If the model includes dir structure, the entire dir structure will be serialized also. Also, serialized data is bound to specific classes.
# torch.save(model, PATH)

# Model class be must be defined somewhere and then
# model = torch.load(PATH) # to load the model
# model.eval() # and evaluate

# 2. Preferred method: 
#If we just want to save our tranined model and use it later for inference, then we can only save the paramaters instead
# of saving the entire model. Any dictionary can be saved with torch.save(). 
#torch.save(model.state_dict(), PATH) # first param will hold the PARAMS of the model.
# to load the model, model must be created again with parameters like following
# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH)) # note that state_dict will take the LOADED dictionary as input
# model.eval() # and evaluate

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn. Linear(n_input_features, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# 1. Lazy method implementation
lazy_model = Model(n_input_features = 6)
# train your model. If you don't train, model will be initialized with some random parameters.
FILE = "lazy_model.pth" # don't need to specificy the extension but its just std practice to show its pytorch model
# torch.save(lazy_model, FILE)

# loading the save model
lazy_model = torch.load(FILE)
lazy_model.eval()

for param in lazy_model.parameters():
    print(f"---------\nNormal model params: {param}\n--------")

# 2. Preferred saving method implementation
# saving
FILE = "preferred_model.pth"
preferred_model = Model(n_input_features = 6)
print (f"-------\ncheck how state_dict looks like: {preferred_model.state_dict()}\n ------")
torch.save(preferred_model.state_dict(), FILE)
#loading
loaded_model = Model(n_input_features=6)

loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(f"------\nLoaded model params: {param}\n-------")

# saving a whole checkpoint during training
FILE = "preferred_model.pth"

checkpoint_model = Model(n_input_features = 6)
learning_rate = 0.001
optimizer = torch.optim.SGD(checkpoint_model.parameters(), lr = learning_rate)
print (f"-------\nCheckpoint saving: check out how optimizer state_dict looks like: {optimizer.state_dict()}\n ------")

checkpoint = {
    "epoch": 90,
    "model_state": checkpoint_model.state_dict(),
    "optimizer_state": optimizer.state_dict()

}
#checkpoint has to be a dictionary
#torch.save(checkpoint, "checkpoint1.pth")

loaded_checkpoint = torch.load("checkpoint1.pth")
epoch = loaded_checkpoint["epoch"]

loaded_checkpoint_model = Model(n_input_features=6)
optimizer =  torch.optim.SGD(checkpoint_model.parameters(), lr = 0) # note lr = 0

loaded_checkpoint_model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
print(f"-------\nCheckpoint saving: check out how loaded checkpoint optimizer state_dict looks like: {optimizer.state_dict()} \n -------")
# note that in the above print the optimizer lr is previous lr. This shows that its value is not lr = 0
# and checkpoint state dict has loaded succesfully

# Saving model on GPU and loading it on CPU ( below is just pseudocode, so it wont work)
device = torch.device('cuda')
gpu_model.to(device)
torch.save(gpu_model.state_dict(). PATH)

device = torch.device('cpu')
gpu_model = Model(*args, **kwargs)
gpu_model.load_state_dict(torch.load(PATH, map_location=device))

# Saving model on GPU and loading it on GPU again
device = torch.device('cuda')
gpu_model.to(device)
torch.save(preferred_model.state_dict(). PATH)

gpu_model = Model(*args, **kwargs)
gpu_model.load_state_dict(torch.load(PATH))
gpu_model.to(device)

# Saving model on CPU but loading it on CPU
torch.save(cpu_model.state_dict(). PATH)

device = torch.device('cuda')
cpu_model = Model(*args, **kwargs)
cpu_model.load_state_dict(torch.load(PATH, map_location="cuda:0")) #cuda:0 when you have multiple GPU and you want it on a specific one
model.to(device)  