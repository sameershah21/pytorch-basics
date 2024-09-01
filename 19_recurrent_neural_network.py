# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter

# RNN allow previous output to be used as input while having hidden states
# they allow us the operate over sequences of vectors
# Pros:
# possiblity to process input of any length
# Model size does not increase with input size
# Computation takes into account historical info
# Weights are shared across time
# Cons:
# Computation is slow
# difficult to get older info
# cannot take into accoount future input of current state
# 
#One hot encoding - One hot encoding vector is filled with 0's except for 1 at index of the current letter 
# One-hot encoding is a method used to represent categorical data as binary vectors.
# Each category (in this case, each letter) is represented by a vector of zeros with a single 1 at the position corresponding to that category.

# Let's use the word "APPLE" as an example for one-hot encoding.

# First, we assign an index to each unique letter in the word.
# Alphabet (considering unique letters in "APPLE"): ['A', 'P', 'L', 'E']

# The index mapping could be:
# 'A' -> 0
# 'P' -> 1
# 'L' -> 2
# 'E' -> 3

# Now, let's create a one-hot encoding for each letter:

# 'A' would be represented as: [1, 0, 0, 0]
# 'P' would be represented as: [0, 1, 0, 0]
# 'L' would be represented as: [0, 0, 1, 0]
# 'E' would be represented as: [0, 0, 0, 1]

# Even though 'A' appears twice in "APPLE", each occurrence of 'A' is encoded the same way.

# So, the one-hot encoding for the word "APPLE" would be a list of these vectors:
# [
#    [1, 0, 0, 0],  # 'A'
#    [0, 1, 0, 0],  # 'P'
#    [0, 1, 0, 0],  # 'P'
#    [0, 0, 1, 0],  # 'L'
#    [0, 0, 0, 1]   # 'E'
# ]

# Notice that the two 'P's in "APPLE" both have the same one-hot encoded vector [0, 1, 0, 0],
# and the 'A' at the beginning has the same encoding as if it appeared elsewhere in the word.

# One-hot encoding doesn't account for the position or repetition of characters;
# it simply represents each unique letter with its own binary vector.



import torch
import torch.nn as nn
import matplotlib.pyplot as plot
from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example

class RNN(nn.Module): # pytorch alredy has RNN module but we implement it here for understanding
    def __init__(self, input_size, hidden_size, output_size): #hidden size is hyperparameter
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # i2h = input to hidden, input = input + hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # i2o = input to output, input = input + hidden
        self.softmax = nn.LogSoftmax(dim=1) # our input function will be of tensor size [1, 57] and we need 2nd dimension
    
    def forward(self, input_tensor, hidden_tensor):
        # refer to last page of RNN pdf to understand forward function code properly
        combined = torch.cat((input_tensor, hidden_tensor), 1) # 1 is dimension
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):# need initial hidden stage in the beginning
        return torch.zeros(1, self.hidden_size)

category_lines, all_possible_categories = load_data () # country as key, name as values
n_categories = len(all_possible_categories)
print (f"No of countries: {n_categories}")

n_hidden = 128 #n_hidden is hyperparameter that we are defining
rnn = RNN(N_LETTERS, n_hidden, n_categories)

#one step as an example (one letter)
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden() # need initial hidden stage in the beginning

output, next_hidden = rnn(input_tensor, hidden_tensor)
print (f"Output size (letter): {output.size()}")    
print (f"Next hidden size (letter): {next_hidden.size()}")


# whole sequence (whole name)
input_tensor = line_to_tensor('Albert')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor) # input_tensor[0] =  means use the first letter
print (f"Output size (name): {output.size()}")    
print (f"Next hidden size (name): {next_hidden.size()}")

def category_from_output(output):
    #likelihood of each character of each category, hence we return index of greatest value
    category_index = torch.argmax(output).item() # single value
    return all_possible_categories[category_index]

print(f"Likelihood of the country from which the name Albert came from is: {category_from_output(output)}")


# loss and optimizer
criterion = nn.NLLLoss() # negaitve likely hood loss
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate)

# train model

def train(line_tensor, category_tensor): #line_tensor = name, category_tensor = index of class label
    '''
    helper function for training 
    '''
    hidden = rnn.init_hidden() # need initial hidden stage in the beginning

    for i in range(line_tensor.size()[0]): # len of the name
        output, hidden = rnn(line_tensor[i], hidden) # line_tensor[i] = current character, hidden = previous hidden state which will be assigned to new hidden state
        # also here output will be the final character
        loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    #get random training sample
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_possible_categories) #line = name

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss
    if (i + 1) % plot_steps == 0: # every 1000 steps
        all_losses.append(current_loss / plot_steps)
        current_loss = 0 # reset current loss so as not to add them to all losses

    if (i + 1) % print_steps == 0: # every 5000 steps
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print (f"Current iteration: {i} {i / n_iters * 100} Current loss: {loss:.4f} Current name {line} \
              Guess: {guess} Is guess Correct? {correct}")


plot.figure()
plot.plot(all_losses)
plot.show()

# at this point we can save our model and use it later or some other places. but lets continue for now
def predict(input_line):
    print(f"\n {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden() # need initial hidden stage in the beginning

        for i in range(line_tensor.size()[0]): # len of the name
            output, hidden = rnn(line_tensor[i], hidden) # line_tensor[i] = current character, hidden = previous hidden state which will be assigned to new hidden state
            # also here output will be the final character
    
    guess = category_from_output(output) # get output from last step
    print(f"Guess: {guess}")
        
while True:
    sentence = input("Input:")
    if sentence == "quit":
        break

    predict(sentence)
