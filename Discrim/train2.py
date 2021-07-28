import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from params import TREE_SIZE, WORKING_SETS, EMBEDDING_SIZE, HIDDEN_SIZE, FEATURES
from net import Net
from net2 import Discrim
from random import choices
import copy



SAMPLE_SIZE = 1

def nMax(l, n):

  return (-l).argsort()[:n]

def sample(l):
  output = []
  probs = np.exp(l)
  x = np.arange(0,TREE_SIZE)
  return choices(x, probs, k=SAMPLE_SIZE)

def sample_noise():
  return np.random.randint(TREE_SIZE, size = SAMPLE_SIZE)

model = Discrim()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)






optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)
loss_func = torch.nn.BCEWithLogitsLoss()


data = pickle.load(open("data/trainingData.pkl", "rb"))
training_data = data['Seq']

seq = []



for i in range(len(training_data)-FEATURES):
  seq.append(training_data[i:i+FEATURES])

data = pickle.load(open("data/testingData.pkl", "rb"))
testing_data = data['Seq']

tseq = []

for i in range(len(testing_data)-FEATURES):
  tseq.append(testing_data[i:i+FEATURES]) 


refmodel = Net(EMBEDDING_SIZE, HIDDEN_SIZE, TREE_SIZE)
refmodel.load_state_dict(torch.load('models/partial1model4.tch'))


def trainDiscrim(inputs, targets):
  inputs= inputs.to(device)
  targets= targets.to(device)
  correct = 0
  for i in range(len(inputs)):
    #print(type(inputs[i]))
    output = model(inputs[i].float())
    #print(output)
    if output >= 0.5 and targets[i] == 1.0:
      correct += 1
    if output < 0.5 and targets[i] == 0.0:
      correct += 1
    loss = loss_func(output[0], targets[i])
    loss.backward()
    #print(loss.grad)
    optimizer.step()
    optimizer.zero_grad()
  return correct

def testDiscrim(inputs, targets):
  correct = 0
  for i in range(len(inputs)):
    output = model(inputs[i])
    if output >= 0.5 and targets[i] == 1.0:
      correct += 1
    if output < 0.5 and targets[i] == 0.0:
      correct += 1
  return correct


for epoch in range(10): 
  print('Epoch: %d' % (epoch))
  correct = 0
  for i in range(len(seq)):
    #model.zero_grad()

    predicted_distribution = refmodel(torch.from_numpy(seq[i]).type(torch.LongTensor))
    predicted_distribution = predicted_distribution.detach().numpy()
    s = sample(predicted_distribution[0])
    s_noise = sample_noise()

    inputs = []
    noise = []

    for j in range(len(s)):
      pfx = np.append(seq[i],[s[j]])
      inputs.append(pfx)
      noisepfx = np.append(seq[i],[s_noise[j]])
      noise.append(noisepfx)

    inputs = np.array(inputs)
    noise = np.array(noise)

    inputs = torch.from_numpy(inputs).type(torch.LongTensor)
    noise = torch.from_numpy(noise).type(torch.LongTensor)
    
    targets = np.ones(len(s))
    targets = torch.from_numpy(targets).float()

    noise_targets = np.zeros(len(s))
    noise_targets = torch.from_numpy(noise_targets).float()

    correct += trainDiscrim(inputs, targets)
    correct += trainDiscrim(noise, noise_targets)


    
    if i % 100 == 0:
      print('[%d], i = %d, Accuracy = %.3f' % (epoch, i/100, correct/(100*2*SAMPLE_SIZE)))
      correct = 0

  print('End of Epoch %d. TESTING' % (epoch))
  with torch.no_grad():
    correct = 0
    for i in range(len(tseq)):
      predicted_distribution = refmodel(torch.from_numpy(tseq[i]).type(torch.LongTensor))
      predicted_distribution = predicted_distribution.detach().numpy()
      s = sample(predicted_distribution[0])
      s_noise = sample_noise()

      inputs = []
      noise = []

      for j in range(len(s)):
        pfx = np.append(tseq[i],[s[j]])
        inputs.append(pfx)
        noisepfx = np.append(tseq[i],[s_noise[j]])
        noise.append(noisepfx)

      inputs = np.array(inputs)
      noise = np.array(noise)

      inputs = torch.from_numpy(inputs).type(torch.LongTensor)
      noise = torch.from_numpy(noise).type(torch.LongTensor)
      
      targets = np.ones(len(s))
      targets = torch.from_numpy(targets).float()

      noise_targets = np.zeros(len(s))
      noise_targets = torch.from_numpy(noise_targets).float()

      correct += testDiscrim(inputs, targets)
      correct += testDiscrim(noise, noise_targets)

      
      if i % 100 == 0:
        print('[TESTING %d], i = %d, Accuracy %d' % (epoch, i/100, correct/(100*2*SAMPLE_SIZE)))
        correct = 0



  torch.save(model.state_dict(), "models/full1model" + str(epoch) + ".tch")



