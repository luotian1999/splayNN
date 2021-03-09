import torch
import torch.nn as nn
import torch.nn.Functional as F
import sys
import numpy as np

class Node:
	def  __init__(self, data):
		self.data = data
		self.parent = None
		self.left = None
		self.right = None
    self.size = 1
    self.height = 1

class SplayTree:
	def __init__(self):
		self.root = None
	
	def __search_tree_helper(self, node, key):
		if node == None or key == node.data:
			return (1,node)

		if key < node.data:
      (cost, node) = self.__search_tree_helper(node.left, key)
			return (cost+1, node)
    (cost, node) = self.__search_tree_helper(node.right, key)
		return (cost+1, node)


	# rotate left at node x
	def __left_rotate(self, x):
		y = x.right
		x.right = y.left
		if y.left != None:
			y.left.parent = x

		y.parent = x.parent
		if x.parent == None:
			self.root = y
		elif x == x.parent.left:
			x.parent.left = y
		else:
			x.parent.right = y
		y.left = x
		x.parent = y

	# rotate right at node x
	def __right_rotate(self, x):
		y = x.left
		x.left = y.right
		if y.right != None:
			y.right.parent = x
		
		y.parent = x.parent
		if x.parent == None:
			self.root = y
		elif x == x.parent.right:
			x.parent.right = y
		else:
			x.parent.left = y
		
		y.right = x
		x.parent = y

	# Splaying operation. It moves x to the root of the tree
	def __splay(self, x):
		while x.parent != None:
			if x.parent.parent == None:
				if x == x.parent.left:
					# zig rotation
					self.__right_rotate(x.parent)

				else:
					# zag rotation
					self.__left_rotate(x.parent)
        x.parent.size = x.parent.right.size + x.parent.left.size + 1
        x.parent.height = max(x.parent.right.height, x.parent.left.height) + 1
        x.size = x.right.size + x.left.size + 1
        x.height = max(x.right.height, x.left.height) + 1
			elif x == x.parent.left and x.parent == x.parent.parent.left:
				# zig-zig rotation
				self.__right_rotate(x.parent.parent)
				self.__right_rotate(x.parent)
        x.parent.parent.size = x.parent.parent.right.size + x.parent.parent.left.size + 1
        x.parent.parent.height = max(x.parent.parent.right.height, x.parent.parent.left.height) + 1
        x.parent.size = x.parent.right.size + x.parent.left.size + 1
        x.parent.height = max(x.parent.right.height, x.parent.left.height) + 1
        x.size = x.right.size + x.left.size + 1
        x.height = max(x.right.height, x.left.height) + 1
			elif x == x.parent.right and x.parent == x.parent.parent.right:
				# zag-zag rotation
				self.__left_rotate(x.parent.parent)
				self.__left_rotate(x.parent)
        x.parent.parent.size = x.parent.parent.right.size + x.parent.parent.left.size + 1
        x.parent.parent.height = max(x.parent.parent.right.height, x.parent.parent.left.height) + 1
        x.parent.size = x.parent.right.size + x.parent.left.size + 1
        x.parent.height = max(x.parent.right.height, x.parent.left.height) + 1
        x.size = x.right.size + x.left.size + 1
        x.height = max(x.right.height, x.left.height) + 1
			elif x == x.parent.right and x.parent == x.parent.parent.left:
				# zig-zag rotation
				self.__left_rotate(x.parent)
				self.__right_rotate(x.parent)
        x.parent.parent.size = x.parent.parent.right.size + x.parent.parent.left.size + 1
        x.parent.parent.height = max(x.parent.parent.right.height, x.parent.parent.left.height) + 1
        x.parent.size = x.parent.right.size + x.parent.left.size + 1
        x.parent.height = max(x.parent.right.height, x.parent.left.height) + 1
        x.size = x.right.size + x.left.size + 1
        x.height = max(x.right.height, x.left.height) + 1
			else:
				# zag-zig rotation
				self.__right_rotate(x.parent)
				self.__left_rotate(x.parent)
        x.parent.parent.size = x.parent.parent.right.size + x.parent.parent.left.size + 1
        x.parent.parent.height = max(x.parent.parent.right.height, x.parent.parent.left.height) + 1
        x.parent.size = x.parent.right.size + x.parent.left.size + 1
        x.parent.height = max(x.parent.right.height, x.parent.left.height) + 1
        x.size = x.right.size + x.left.size + 1
        x.height = max(x.right.height, x.left.height) + 1
  


	# search the tree for the key k
	# and return the corresponding node
	def access(self, k):
		(cost, x) = self.__search_tree_helper(self.root, k)
    size = -1
    height = -1
		if x != None:
      size = x.size
      height = x.height
			self.__splay(x)
    return (size, height, cost)

  def access_no_splay(self, k):
		(cost, x) = self.__search_tree_helper(self.root, k)
    size = -1
    height = -1
		if x != None:
      size = x.size
      height = x.height
    return (size, height, cost)
  

	# insert the key to the tree in its appropriate position
	def insert(self, key):
		node =  Node(key)
		y = None
		x = self.root

		while x != None:
			y = x
			if node.data < x.data:
				x = x.left
			else:
				x = x.right

		# y is parent of x
		node.parent = y
		if y == None:
			self.root = node
		elif node.data < y.data:
			y.left = node
		else:
			y.right = node
		# splay the node
		self.__splay(node)





# define Network

class net(nn.Module):
  def __init__(self, input_size=200, output_size=2, hidden_size=16):
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size

    self.fc1 = nn.Linear(self.input_size, self.hidden_size)
    self.fc2 = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, x):
    out = self.fc1(x)
    out = F.relu(out)
    out = self.fc2(out)
    return out

# define loss function
def tian_loss(output, should_splay, cost):
  # TODO: SIMULATE
  t = [1,0] if should_splay else [0,1]
  targets = torch.Tensor(t)


  # TODO: check my implementation of this!
  return torch.sum(- cost * targets * F.log_softmax(output, -1), -1)
  
  

# prepare data + other setup
# inputs = ...
PATH = "./model_test.tch" #CHANGE THIS
model = net()
optimizer = optim.SGD(model.parameters())
running_loss = 0.0


#initialize Splay tree here:
tree = SplayTree()
#insert nodes below:
# TODO: generate seq
# seq = ..

# training loop
for i in range(10000): # 10 000 = fake size of data
  # TODO: generate new inputs 
  
  yestree = copy.deepcopy(tree)
  notree = copy.deepcopy(tree)

  # simulate ytree
  yshc = [yestree.access(seq[i+j]) for j in range(0, 100)]
  yunzipped_shc = list(zip(*yshc))

  ysize = torch.from_numpy(np.asarray(yunzipped_shc[0]))
  yheight = torch.from_numpy(np.asarray(yunzipped_shc[1]))
  ycost = sum(yunzipped_shc[2])

  # simulate ntree
  nshc = [ (notree.access_no_splay(seq[i+j]) if j == 0 else notree.access(seq[i+j])) for j in range(0, 100)]
  nunzipped_shc = list(zip(*nshc))

  nsize = torch.from_numpy(np.asarray(nunzipped_shc[0]))
  nheight = torch.from_numpy(np.asarray(nunzipped_shc[1]))
  ncost = sum(nunzipped_shc[2])

  if (ycost > ncost):
    should_splay = False
  else:
    should_splay = True
  
  cost = abs(ycost - ncost)
  
  # train
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = tian_loss(outputs, should_splay, cost) 
  loss.backward() # torch autograd should be able to define backward for you automatically
  optimizer.step()

  # print statistics
  running_loss += loss.item()
  if i % 10 == 0:    # TODO: change 10 to sth else, print running loss every 10 items
    print('[%d] loss: %.3f' %
                  (i + 1, running_loss / 10))
    running_loss = 0.0

  #save network
  torch.save(model.state_dict(), PATH) 


