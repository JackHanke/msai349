from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  # takes array arr of frequencies and calculates entropy
  def entropy_calc(arr, base=2): 
    # helper function to avoid math error for log(0)
    def entropy_term(freq, base):
      if freq in (0,1): return 0
      return -1*freq*math.log(freq,base)
    return sum([entropy_term(freq, base=base) for freq in arr])
  
  # creates possible values dictionary of each attributes and class
  possible_vals = {}
  for row in examples:
    for key, val in row.items():
      try:
        if val not in possible_vals[key]:
          possible_vals[key].append(val)
      except KeyError:
        possible_vals[key] = [val]
  for key in possible_vals: possible_vals[key].sort() # TODO so ratchet

  root_node = Node()
  
  attributes = [att for att in examples[0].keys() if att != 'Class']

  best_att = None
  smallest_entropy = 1
  for att in attributes:
    avg_entropy = 0
    for att_val in possible_vals[att]:
      relative_freqs = []
      denom = len([1 for row in examples if row[att] == att_val])
      for class_val in possible_vals['Class']:
        relative_freqs.append(len([1 for row in examples if row[att] == att_val and row['Class'] == class_val])/denom)
      att_entropy = entropy_calc(relative_freqs)
      avg_entropy += att_entropy*(denom/len(examples))
      # print(f'attribute is {att}')
      # print(f'attribute value = {att_val}')
      # print(f'denom = {denom}')
      # print(relative_freqs)
      # print(f'entropy of relative freqs = {att_entropy}')
      # print(f'component of avg_entropy = {(denom/len(examples))}')
    if avg_entropy < smallest_entropy: # this breaks ties by consistently picking the first attribute with lowest entropy
      smallest_entropy = avg_entropy
      best_att = att
    print(f'attribute is {att}')
    print(avg_entropy)

  print(best_att)
  input()


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

if __name__ == '__main__':
  from parse import parse
  examples = parse('tennis.data')
  default = 0
  ID3(examples=examples, default=default)
