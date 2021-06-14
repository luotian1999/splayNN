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
from splaytree import Tree
from net import SplayNet, NN_INPUT_NODES, FEATURES_PER_INPUT_NODE, TREE_SIZE

TEST_WORKING_SET_DURATION = 1000
TEST_WORKING_SET_TIMES = 200
PATH = "model_LCA_seqdepth.tch"

model = SplayNet()
model.load_state_dict(torch.load(PATH))

#generate seq (dynamic zipf)
seq = np.zeros(TEST_WORKING_SET_DURATION * TEST_WORKING_SET_TIMES)
x = np.arange(TREE_SIZE)
weights = np.ones(TREE_SIZE)
weights /= weights.sum()
zipf = stats.rv_discrete(name='zipf', values=(x, weights))

for i in range(TEST_WORKING_SET_TIMES):
  perm = np.random.permutation(TREE_SIZE)
  for j in range(TEST_WORKING_SET_DURATION):
    z = zipf.rvs()
    seq[i * TEST_WORKING_SET_DURATION + j] = perm[z]

tree = Tree(TREE_SIZE)
control_tree = Tree(TREE_SIZE)

inputs = torch.zeros(FEATURES_PER_INPUT_NODE * NN_INPUT_NODES)


with torch.no_grad():
  total_depth = 0
  control_depth = 0
  for i in range(NN_INPUT_NODES - 1, TEST_WORKING_SET_DURATION * TEST_WORKING_SET_TIMES):
    d = tree.depth(seq[i])
    for k in range(NN_INPUT_NODES):
      #inputs[k] = seq[i-k]
      inputs[k] = tree.depth(seq[i-k])

      #last set of inputs is the distance between the node from k accesses ago 
      #and the least common ancestor of that node and the current node in T_i
      inputs[1 * NN_INPUT_NODES + k] = d - tree.lca(seq[i-k], seq[i])
    
    output = model(inputs)
    res = torch.argmax(output, dim=0)

    if res == 1:
      total_depth += tree.access_splay(seq[i])[1]
    else: 
      total_depth += tree.access(seq[i])[1]
    control_depth += control_tree.access_splay(seq[i])[1]

    if i % 1000 == 0:
      print('[i = %d], total_depth = %d, control_depth = %d' 
      % (i, total_depth, control_depth))
      #print(inputs)

print('FINAL: total_depth = %d, control_depth = %d' % (total_depth, control_depth))


