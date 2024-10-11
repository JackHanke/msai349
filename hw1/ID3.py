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
    if avg_entropy < smallest_entropy: # this breaks ties by consistently picking the first attribute with lowest entropy
      smallest_entropy = avg_entropy
      best_att = att
  node = Node(label=best_att, children={})
  input()

  return node


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  while len(node.children.keys()) > 0: # if we are not at leaf node
    value = example[node.label] # value of attribute at node
    node = node.children[value] # go to the child at 
  return example[node.label]


def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  correct_total = 0
  total = len(examples)
  for row in examples:
    prediction = evaluate(node=node, example=row)
    if prediction == row['Class']: correct_total += 1
  return correct_total/total


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''


if __name__ == '__main__':
  from parse import parse
  examples = parse('tennis.data')
  default = 0 # TODO wtf is default
  root_node = ID3(examples=examples, default=default)

  # make prediction on row of data
  row = examples[0]
  prediction = evaluate(node=root_node, example=row)
  print(f'Prediction for row: {row} is equal to {prediction}')

  # test tree on full dataset
  accuracy = test(node=root_node, examples=examples)
  print(f'Accuracy of tree = {accuracy:.4f}')
