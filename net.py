import torch
import torch.nn as nn

TREE_SIZE = 1000
NN_INPUT_NODES = 20
FEATURES_PER_INPUT_NODE = 2

class SplayNet(nn.Module):
  def __init__(self, input_size= FEATURES_PER_INPUT_NODE * NN_INPUT_NODES, output_size=2, hidden_size=16):
    super(SplayNet, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    #self.hidden_size = hidden_size

    #self.fc1 = nn.Linear(self.input_size, self.hidden_size)
    #self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    self.fc = nn.Linear(self.input_size, self.output_size)

  def forward(self, x):
    x = self.fc(x)
    return x