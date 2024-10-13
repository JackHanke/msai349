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

  # 
  def find_best_attribute(examples, possible_vals, attributes):
    # initialize best attribute and the corresponding smallest entropy
    best_att = None
    smallest_entropy = float('inf')
    # loop over attributes
    for att in attributes:
      avg_entropy = 0
      # loop over all values for attribute att
      for att_val in possible_vals[att]:
        # relative_freqs is an array of frequencies, denoted p_i in the slides
        relative_freqs = [] 
        # the numerator of the frequency is the number of rows that have the attribute att == att_val and the class value = class_val
        # the denominator (variable denom) of the frequency is the number of rows that have attribute att == att_val
        denom = len([1 for row in examples if row[att] == att_val])
        # loop over all class values
        for class_val in possible_vals['Class']:
          relative_freqs.append(len([1 for row in examples if row[att] == att_val and row['Class'] == class_val])/denom)
        att_entropy = entropy_calc(relative_freqs, base=len(possible_vals['Class']))
        avg_entropy += att_entropy*(denom/len(examples)) # average entropy over attribute values
      # if average entropy is less than previous entropies, set as new lowest entropy
      # this breaks ties by picking the first attribute with lowest entropy
      if avg_entropy < smallest_entropy: 
        smallest_entropy = avg_entropy
        best_att = att
    # TODO is this check necessary?
    if smallest_entropy == float('inf'): return None
    # 
    return best_att, smallest_entropy

  # 
  def tree_build(examples, node, attributes, entropy_threshold):
    # calculate best attribute among attributes to split on for given examples
    best_att, smallest_entropy = find_best_attribute(examples=examples, possible_vals=possible_vals, attributes=attributes)
    print(f'best_att = {best_att}, smallest_entropy = {smallest_entropy}')
    input()
    # set label of node as attribute the node splits on
    node.label = best_att
    # TODO take attribute out that was split ?
    attributes.remove(best_att)

    # 
    for att_val in possible_vals[best_att]:
        node.children[att_val] = Node()

    # TODO remove classified rows from examples
    filtered_examples = []
    for row in examples:
      if row[best_att] == 1: pass

    for att_val, child in node.children.items():
      # 
      if smallest_entropy <= entropy_threshold:
        tree_build(examples=examples, node=child, attributes=attributes, entropy_threshold=entropy_threshold)
    
  # creates dictionary of possible values for each attributes
  possible_vals = {}
  for row in examples:
    for key, val in row.items():
      try:
        if val not in possible_vals[key]:
          possible_vals[key].append(val)
      except KeyError:
        possible_vals[key] = [val]
  for key in possible_vals: possible_vals[key].sort() # TODO better way to do this? kinda ugly

  # generate attributes TODO inside or outside tree loop?
  attributes = [att for att in examples[0].keys() if att != 'Class']

  # TODO write something like this idk
  root_node = Node()
  tree_build(examples=examples, node=root_node, attributes=attributes, entropy_threshold=0)
  return root_node


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
  examples = parse('mushroom.data')
  default = 0 # TODO wtf is default
  root_node = ID3(examples=examples, default=default)

  # make prediction on row of data
  row = examples[0]
  prediction = evaluate(node=root_node, example=row)
  print(f'Prediction for row: {row} is equal to {prediction}')

  # test tree on full dataset
  accuracy = test(node=root_node, examples=examples)
  print(f'Accuracy of tree = {accuracy:.4f}')
