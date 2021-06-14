import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import numpy as np
import copy
import collections
import random

class Node:
  def  __init__(self, data):
    self.data = data
    self.parent = None
    self.left = None
    self.right = None
    self.size = 1
    self.height = 1
    self.LCAset = set()

    self.modified = False

    self.parent_old = None
    self.left_old = None
    self.right_old = None
    self.size_old = None
    self.height_old = None
    self.LCAset_old = None
  
  def __repr__(self):
    return hex(id(self))
  

class Tree:
  def init_balanced_h(self, start, end):
    if start > end: return None
    mid = (start + end) // 2
    curr = Node(mid)
    curr.left = self.init_balanced_h(start, mid - 1)
    curr.right = self.init_balanced_h(mid + 1, end)
    if curr.left != None : curr.left.parent = curr
    if curr.right != None: curr.right.parent = curr
    return curr

  def __init__(self, size):
    self.root = self.init_balanced_h(0, size-1)
    self.old_root = None
    self.cached = []
    self.rotate_up = False 
  
  def __search_tree_helper(self, node, key, remove=False, add=False):
    assert(node != None)

    if add:
      node.LCAset.add(key)
    if remove:
      node.LCAset.remove(key)
    
    if key == node.data:
      return (0,node)

    if key < node.data:
      (cost, node) = self.__search_tree_helper(node.left, key)
      return (cost+1, node)

    (cost, node) = self.__search_tree_helper(node.right, key)
    return (cost+1, node)
  
  def cache(self, x):
    assert(x.modified == False)

    self.cached.append(x)
    x.modified = True

    x.parent_old = x.parent
    x.left_old = x.left
    x.right_old = x.right
    x.size_old = x.size
    x.height_old = x.height
    x.LCAset_old = x.LCAset
  
  def restore(self):
    for x in self.cached:
      assert(x.modified == True)
      x.modified = False

      x.parent = x.parent_old
      x.left = x.left_old
      x.right = x.right_old
      x.size = x.size_old
      x.height = x.height_old
      x.LCAset = x.LCAset_old

      x.parent_old = None
      x.left_old = None
      x.right_old = None
      x.size_old = None
      x.height_old = None
      x.LCAset_old = None
    
    self.cached = []
    self.root = self.old_root
    self.old_root = None
  
  def save(self):
    for x in self.cached:
      assert(x.modified == True)
      x.modified = False    

      x.parent_old = None
      x.left_old = None
      x.right_old = None
      x.size_old = None
      x.height_old = None 
      x.LCAset_old = None
    
    self.cached = []
    self.old_root = self.root

  # rotate left at node x
  def left_rotate(self, x):
    if not x.modified: self.cache(x)
    if x.parent != None:
      if not x.parent.modified: self.cache(x.parent)
    
    y = x.right
    if not y.modified: self.cache(y)
    if y.left != None:
      if not y.left.modified: self.cache(y.left)

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
  def right_rotate(self, x):
    if not x.modified: self.cache(x)
    if x.parent != None:
      if not x.parent.modified: self.cache(x.parent)
    
    y = x.left
    if not y.modified: self.cache(y) 
    if y.right != None:
      if not y.right.modified: self.cache(y.right)

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

  def size(self, x):
    if x == None: return 0
    return x.size
    
  def height(self, x):
    if x == None: return 0
    return x.height

  def LCAset(self, x):
    if x == None: return set()
    return x.LCAset

  def update(self, x):
    x.size = self.size(x.left) + self.size(x.right) + 1
    x.height = max(self.height(x.left), self.height(x.right)) + 1
    x.LCA_set = (self.LCAset(x.left))|(self.LCAset(x.right))

  def updateLCAset(self, x):
      x.LCA_set = (self.LCAset(x.left))|(self.LCAset(x.right))

  # Splaying operation. It moves x to the root of the tree
  def splay(self, x):
    while x.parent != None:
      p = x.parent
      g = x.parent.parent

      if g == None:
        if x == p.left:
          # zig rotation
          self.right_rotate(p)

        else:
          # zag rotation
          self.left_rotate(p)

        self.update(p)
        self.update(x)

      elif x == p.left and p == g.left:
        # zig-zig rotation
        self.right_rotate(x.parent.parent)
        self.right_rotate(x.parent)

        self.update(g)
        self.update(p)
        self.update(x)

      elif x == p.right and p == g.right:
        # zag-zag rotation
        self.left_rotate(x.parent.parent)
        self.left_rotate(x.parent)

        self.update(g)
        self.update(p)
        self.update(x)

      elif x == p.right and p == g.left:
        # zig-zag rotation
        self.left_rotate(x.parent)
        self.right_rotate(x.parent)

        self.update(g)
        self.update(p)
        self.update(x)

      else:
        # zag-zig rotation
        self.right_rotate(x.parent)
        self.left_rotate(x.parent)
        
        self.update(g)
        self.update(p)
        self.update(x)
  
  def depth(self, x):
    return self.access(x)[1]

  def lca(self, x, y):
    m = min(x,y)
    M = max(x,y)

    curr = self.root
    depth_curr = 0
    while not (m <= curr.data and curr.data <= M):
      if M < curr.data: curr = curr.left
      elif curr.data < m: curr = curr.right
      else: assert(False)
      depth_curr += 1
    
    return self.depth(x) - depth_curr

  def rotateup(self, x):
    while x.parent != None:
      p = x.parent
      if x == p.left:
        self.right_rotate(p)

      else:
        assert(x == p.right)
        # zag-zag rotation
        self.left_rotate(p)

      self.update(p)
      self.update(x)

  # search the tree for the key k
  # and return the corresponding node
  def access_splay(self, k):
    (cost, x) = self.__search_tree_helper(self.root, k)
    assert(x != None)
    if self.rotate_up == True: self.rotateup(x)
    else:
      self.splay(x)
    return (x, cost)

  def access(self, k):
    (cost, x) = self.__search_tree_helper(self.root, k)
    return (x, cost)

  def addLCA(self, k):
    (cost, x) = self.__search_tree_helper(self.root, k, add=True)

  def removeLCA(self, k):
    (cost, x) = self.__search_tree_helper(self.root, k, remove=True)
  
  def tprint_h(self, x, depth):
    if x.right != None: self.tprint_h(x.right, depth + 1)
    print('\t' * depth + '%d' % (x.data))
    if x.left != None: self.tprint_h(x.left, depth + 1)

  def tprint(self):
    print('---------------------------------------')
    zz = [x.data for x in self.cached]
    if self.root != None: self.tprint_h(self.root, 0)
    print('---------------------------------------')
  
  def check_h(self, x, checked):
    if(x.data in checked): 
      print("%d in tree twice" % (x.data))
      self.tprint()
      assert False
    checked.add(x.data)
    if(x.parent != None):
      if(not (x.parent.left == x or x.parent.right == x)):
        print("%d failed parent check"% (x.data))
        self.tprint()
        assert False
    if(x.left != None):
      if(x.left.data >= x.data):
        print("x.left = %d, x = %d" % (x.left.data, x.data))
        self.tprint()
        assert False
      self.check_h(x.left, checked)
    if(x.right != None):
      if(x.right.data <= x.data):
        print("x.right = %d, x = %d" % (x.right.data, x.data))
        self.tprint()
        assert False
      self.check_h(x.right, checked)
    if((x in self.cached) != x.modified):
      print("x in self.cached = %i, x.modified = %i" % (x in self.cached, x.modified))
      self.tprint()
      assert False


  def check(self):
    if(self.root == None):
      print("root = None")
      assert False

    m_set = set()
    self.check_h(self.root, m_set)

    assert(len(m_set) == self.root.size)


if __name__ == "__main__":
  tree = Tree(50)
  tree.tprint()

  (x, cost) = tree.access_splay(10)
  print("access splay cost", cost)
  tree.tprint()