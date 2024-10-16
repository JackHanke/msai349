class Node:
  def __init__(self):
    self.label = None
    self.parent = None
    self.children = {}
    self.relative_freqs = None
    self.leaf_eval = None