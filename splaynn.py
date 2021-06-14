import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import numpy as np
import copy
import random
import time
import scipy.stats as stats
import faulthandler; faulthandler.enable()
from net import SplayNet, NN_INPUT_NODES, FEATURES_PER_INPUT_NODE, TREE_SIZE
from splaytree import Tree

sys.setrecursionlimit(1500000)

#information regarding the randomly generated input sequence
#how often we switch the zipf distribution
WORKING_SET_DURATION = 2000
WORKING_SET_TIMES = 30

#for training/determining the weights 
NODES_TO_CHECK = 50

#changes all "splay" to "rotate-ups"
ROTATE_UP = False

# Initialize Model
#PATH = "./model"+time.strftime("%Y%m%d-%H%M%S")+".tch"
PATH = "./model_LCA_seqdepth.tch"
model = SplayNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

running_loss = 0.0

#generate seq (dynamic zipf)
seq = np.zeros(WORKING_SET_DURATION * WORKING_SET_TIMES)
x = np.arange(TREE_SIZE)
weights = np.ones(TREE_SIZE)
weights /= weights.sum()
zipf = stats.rv_discrete(name='zipf', values=(x, weights))

for i in range(WORKING_SET_TIMES):
  perm = np.random.permutation(TREE_SIZE)
  for j in range(WORKING_SET_DURATION):
    z = zipf.rvs()
    seq[i * WORKING_SET_DURATION + j] = perm[z]


#initialize trees
tree = Tree(TREE_SIZE)
tree.rotate_up = ROTATE_UP

control_tree = Tree(TREE_SIZE)
control_tree.rotate_up = ROTATE_UP

print("Successfully initialized trees!")

inputs = torch.zeros(FEATURES_PER_INPUT_NODE * NN_INPUT_NODES)

#performance tracking
false_positives = 0
false_negatives = 0
num_predictions = 0
totalcost = 0
control_totalcost = 0
splaydepths = []
nosplaydepths = []

# training loop
time1 = time.time()

#seqdepths[i] is the depth of i in tree T_i, where T_i is the tree right before the i-th access
seqdepths = []
for i in range(NN_INPUT_NODES - 1):
  seqdepths.append(tree.depth(seq[i]))

for i in range(NN_INPUT_NODES - 1, WORKING_SET_DURATION * WORKING_SET_TIMES  - (NODES_TO_CHECK - 1)):
  seqdepths.append(tree.depth(seq[i]))
  d = tree.depth(seq[i])

  for k in range(NN_INPUT_NODES):
    #inputs[k] = seq[i-k]
    inputs[k] = seqdepths[i-k]
    #inputs[k] = tree.depth(seq[i-k])

    #last set of inputs is the distance between the node from k accesses ago 
    #and the least common ancestor of that node and the current node in T_i
    inputs[1 * NN_INPUT_NODES + k] = d - tree.lca(seq[i-k], seq[i])

  # simulate what happens if we splay (or rotate-up if that's set to True)
  # the next NODES_TO_CHECK accesses
  tree.save()

  yshc = [ (tree.access_splay(seq[i+j]) if j == 0 else tree.access(seq[i+j])) for j in range(NODES_TO_CHECK)]
  #yshc = [tree.access_splay(seq[i+j]) for j in range(NODES_TO_CHECK)]

  #yunzipped_shc = list(zip(*yshc))
  ycost = sum([access[1] for access in yshc])
  tree.restore()

  # simulate what happens if we don't splay/rotate-up the current node, but
  # splay/rotate-up the next NODES_TO_CHECK - 1 accesses
  tree.save()

  nshc = [tree.access(seq[i+j]) for j in range(NODES_TO_CHECK)]
  #nshc = [ (tree.access(seq[i+j]) if j == 0 else tree.access_splay(seq[i+j])) for j in range(NODES_TO_CHECK)]

  #nunzipped_shc = list(zip(*nshc))
  ncost = sum([access[1] for access in nshc])
  tree.restore()


  #sanity check
  #if inputs[0] > 500:
  if ycost < ncost:
    should_splay = 1
    reference_output = torch.from_numpy(np.array([0., 1.])).float()
  else:
    should_splay = 0
    reference_output = torch.from_numpy(np.array([1., 0.])).float()
  
  control_totalcost += control_tree.access_splay(seq[i])[1]

  # train
  optimizer.zero_grad()
  output = model(inputs).float()
  res = torch.argmax(output, dim=0)

  if res == 1:
    totalcost += yshc[0][1]
    tree.access_splay(seq[i])

    splaydepths.append(inputs.numpy()[0])
    if should_splay == 0: false_positives += 1
  else:
    totalcost += nshc[0][1]

    nosplaydepths.append(inputs.numpy()[0])
    if (should_splay == 1): false_negatives += 1
  
  num_predictions += 1

  loss = criterion(output, reference_output)
  loss.backward()
  optimizer.step()

  # print statistics
  if i % 1000 == 0:
    print('[i = %d], false +\'s= %d, false -\'s = %d, acc = %.2f totalcost = %d, control_tc = %d,\
    splaydepths (avg, len) = (%.1f, %d), nosplaydepths (avg, len) = (%.1f, %d)' 
    % (i, false_positives, false_negatives, 1 - (false_negatives + false_positives) / num_predictions, totalcost, control_totalcost,
        -1 if len(splaydepths) == 0 else sum(splaydepths)/len(splaydepths), len(splaydepths),
        -1 if len(nosplaydepths) == 0 else sum(nosplaydepths)/len(nosplaydepths), len(nosplaydepths)))
    #print(inputs)

print("Successfully trained!, time =", time.time() - time1)
torch.save(model.state_dict(), PATH)
print("Successfully saved model to ", PATH)