import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
  def __init__(self, embedding_size, hidden_size, output_size):
    super(Net, self).__init__()
    self.hidden_size = hidden_size

    self.word_embeddings = nn.Embedding(output_size, embedding_size)

    # The LSTM takes word embeddings as inputs, and outputs hidden states
    # with dimensionality hidden_size.
    self.lstm = nn.LSTM(embedding_size, hidden_size, 2)

    # The linear layer that maps from hidden state space to tag space
    self.fc1 = nn.Linear(hidden_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, input):
    embeds = self.word_embeddings(input)
    lstm_out, _ = self.lstm(embeds.view(len(input), 1, -1))
    output1 = self.fc1(lstm_out.view(len(input), -1))
    output1 = F.relu(output1)
    output2 = self.fc2(output1)
    output = F.log_softmax(output2, dim=1)
    return output[-1].view(1, -1)
