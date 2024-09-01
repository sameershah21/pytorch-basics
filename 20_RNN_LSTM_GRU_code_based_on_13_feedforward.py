# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter
# chapter 20 - image classfication isn't typically the best example of RNNS but we want to see is
# how to use input as a sequence and correct shapes, and how RNNs can be used to get high accuracy on classfication problems
# GRU - Gated Recurrent unit
# LSTM - Long Short term memory
# Both are popular RNNs
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plot

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 0 prepare data
# hyper parameters
#input_size = 784 # 28x28=724 image size. we will flatten to 1d tensor
input_size = 28 # we want to make one image dimension as one sequence and another image dimension as input or feature size
sequence_length = 28
hidden_size = 128
num_layers = 2 # by default RNN is one layer. when value is 2, second RNN layer's input will be be first RNN layers's output. this is done to improve the model.

num_classes = 10
num_epochs = 2 # set to higher value and it will take more time
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# take a look at 1 batch 
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)
# samples output: torch.Size([100, 1, 28, 28]) - 100 = batch size, 1 = just 1 channel (no colors), 28x28 image size
# labels output : torch.Size([100]) for each class label we have one value

for i in range(6):
    plot.subplot(2, 3, i+1)
    plot.imshow(samples[i][0], cmap='gray') # to display the data [0] is to access the first channel
plot.show()

# Step 1 define model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)#use built in pytorch RNN/GRU/LSTM module. Input needs to have shape batch size, sequence length and input size
        # x -> (batch_size, seq, input_size) 
        # IMPORTANT: once done with RNN. replace RNN with GRU, and LSTM and try. for LSTM we need initial cell state (c0_for_LSTM)
        self.fc= nn.Linear(hidden_size,num_classes) # create fully connected linear layer # 
        #as per the diagram of sentiment classfication ( slide 4 of RNN.pdf), we just need the list time step of the sequence ot the classfication,
        # so we only need the last hidden layer as input size  

    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  #initial hidden state
        c0_for_LSTM = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # out, _ = self.RNN(x, h0) #
        # out, _ = self.GRU(x, h0) #
        out, _ = self.LSTM(x, (h0, c0_for_LSTM)) # _ means don't need that part
        # new output shape is - batch_size, sequence_length, hidden_size
        #out (N, 28, 128)
        out = out[:, -1, :] # need to reshape before passing to fully connected layer. 
        #Hence we slice - take all the sample in batch(:), only last time step (-1), take all the features in hidden size(:)
        #only decode the hidden state of the last time step i.e out (N, 128)

        out = self.fc(out)
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device) # these values are hyper parameters
#
# A hyperparameter is a parameter used to control the training process of a machine learning model.
#  Unlike model parameters, which are learned from the data during training, hyperparameters are set before
#  the training process begins and remain fixed during training.

# Step 2 loss and optimzer
criterion = nn.CrossEntropyLoss() # applies softmax
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Step 3 training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader): # enum will give index (i) and data (tuple of images and labels). (remembers enum gives indes as well as corresponding value. useful when you want to keep track of data) 
        #shape is 100, 1, 28, 28 
        #input size 784, image tensor needs 100,28,28
        # hence we need to reshape
        images = images.reshape(-1, sequence_length , input_size).to(device) # remember putting -1 will tell torch to calculate the remaining value(dimension) automatically
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs,labels)

        #backward
        optimizer.zero_grad() # remember grad issue reset in computational as per previous chapters
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0: # every 100th step do the following
            print (f"epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss:.4f}")

# Step 4 testing and evaluation

with torch.no_grad(): # don't want to compute gradient for all the steps
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        #reshape again
        images = images.reshape(-1, sequence_length, input_size).to(device) # remember putting -1 will tell torch to calculate the remaining value(dimension) automatically
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1) # will give outputs along with dimension which is 1 which is index. we don't care about values hence we put _ to ignore it
        n_samples += labels.shape[0] # will give no. of samples in current batch (100)
        n_correct += (predictions == labels).sum().item() # for each correct prediction we add +1
    
    acc = 100.9 * n_correct / n_samples

    print(f"accuracy: {acc}")

