import torch
import torch.nn as nn
import torch.nn.functional as F

TREE_SIZE = 1000
NN_INPUT_NODES = 20
FEATURES_PER_INPUT_NODE = 1

class Discrim(nn.Module):
  def __init__(self, input_size= 2, output_size=1, hidden_size=4):
    super(Discrim, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size

    self.fc1 = nn.Linear(self.input_size, self.hidden_size)
    self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    #self.fc = nn.Linear(self.input_size, self.output_size)

  def forward(self, x):
    #x = self.fc(x)
    
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)

    return x
