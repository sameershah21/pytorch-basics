# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter
# pytorch lightning is a wrapper which makes it easy to write code in pytorch by providing addtional functions
#  conda install pytorch-lightning -c conda-forge 
# this  tutorial could not be completed properly due to version changes of lightning

import torch
import torch.nn as nn
import torch.utils

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plot
import pytorch_lightning as pl
import torch.nn.functional as F
import os
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning import Trainer



# Step 0 prepare data
# hyper parameters
input_size = 784 # 28x28=724 image size. we will flatten to 1d tensor
hidden_size = 100
num_classes = 10
num_epochs = 2 # set to higher value and it will take more time
batch_size = 100
learning_rate = 0.001



# Step 1 define model
class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size) # layer 1
        self.relu = nn.ReLU() # activation function layer 2
        self.linear2 = nn.Linear(hidden_size, num_classes) # layer 3 num_classes is output size. for each possible class we have one output
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax in the end since this uses pytorch cross entropy loss. If that wasn't the case, we would have to calculate softmax
        return out
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        images, labels = batch    
        images = images.reshape(-1, 28*28) # remember putting -1 will tell torch to calculate the remaining value(dimension) automatically        

        # forward
        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)   
        tensorboard_logs =  {'train_loss': loss}
        return {'loss': loss, 'log':  tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)     
        return optimizer    

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # before first run  and lightning's suggestion
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=batch_size,
                                                   num_workers= 4,
                                                    shuffle=True) # afte the first run and lightning's suggestion
        return train_loader
    
    def validation_step(self, batch, batch_idx): #note that function name should be exactly the same as what lightning requires
        # training_step defines the train loop. It is independent of forward
        images, labels = batch    
        images = images.reshape(-1, 28*28) # remember putting -1 will tell torch to calculate the remaining value(dimension) automatically        

        # forward
        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)   
        return {'val_loss': loss}

    def val_dataloader(self): #note that function name should be exactly the same as what lightning requires
        val_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=False)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # before first run  and lightning's suggestion
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=batch_size,
                                                num_workers= 4,
                                                shuffle=False)
        return val_loader
    
    # def validation_epoch_end(self, outputs):
    #     '''
    #     this function is executed after each validation epoch
    #     '''
    #     # outputs = list of dictionaries
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

    #     # use key 'log'
    #     return {'val_loss': avg_loss}
    
if __name__ == '__main__':
    #trainer = Trainer(fast_dev_run=False)# fast dev run true will run single batch thru training and validation. This is very helpful during development where you dont need to iterate over all the first batch manually to get its value
    # trainer = Trainer(auto_lr_find=True, fast_dev_run=True) # find best learning rate but seems like this function and gpus and many others are not there anymore
    trainer = Trainer(max_epochs= num_epochs, fast_dev_run=False)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)

# you can see that lightning_logs dir gets created automatically
# because of lighting we:
# 1. no longer need to have to device syntax
# 2. no need to for loop epoch and all the batches
# 3. no need to define optimzer and backward pass
# 4. no need to manually install tensorboard because its comes inbuilt