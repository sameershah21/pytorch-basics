# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

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
input_size = 784 # 28x28=724 image size. we will flatten to 1d tensor
hidden_size = 100
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
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size) # layer 1
        self.relu = nn.ReLU() # activation function layer 2
        self.linear2 = nn.Linear(hidden_size, num_classes) # layer 3 num_classes is output size. for each possible class we have one output
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax in the end since this uses pytorch cross entropy loss. If that wasn't the case, we would have to calculate softmax
        return out

model = NeuralNet2(input_size=28*28, hidden_size=100, num_classes=10).to(device)

# Step 2 loss and optimzer
criterion = nn.CrossEntropyLoss() # applies softmax
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Step 3 training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader): # enum will give index (i) and data (tuple of images and labels). (remembers enum gives indes as well as corresponding value. useful when you want to keep track of data) 
        #shape is 100, 1, 28, 28 
        #input size 784, image tensor needs 100,784
        # hence we need to reshape
        images = images.reshape(-1, 28*28).to(device) # remember putting -1 will tell torch to calculate the remaining value(dimension) automatically
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
        images = images.reshape(-1, 28*28).to(device) # remember putting -1 will tell torch to calculate the remaining value(dimension) automatically
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1) # will give outputs along with dimension which is 1 which is index. we don't care about values hence we put _ to ignore it
        n_samples += labels.shape[0] # will give no. of samples in current batch (100)
        n_correct += (predictions == labels).sum().item() # for each correct prediction we add +1
    
    acc = 100.9 * n_correct / n_samples

    print(f"accuracy: {acc}")

