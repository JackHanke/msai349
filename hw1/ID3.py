from node import Node
import math

def ID3(examples, default, possible_vals=None):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples. Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

  # Takes array arr of frequencies and calculates entropy
  def entropy_calc(arr, base=2): 
    # Helper function to avoid math error for log(0)
    def entropy_term(freq, base):
      if freq == 0 or freq == 1:
        return 0
      return -freq * math.log(freq, base)
    return sum([entropy_term(freq, base=base) for freq in arr])

  # Finds the best attribute to split on
  def find_best_attribute(examples, possible_vals, attributes):
    # Initialize best attribute and the corresponding smallest entropy
    best_att = None
    smallest_entropy = float('inf')
    # Loop over attributes
    for att in attributes:
      avg_entropy = 0
      # Loop over all values for attribute att
      for att_val in possible_vals[att]:
        # Relative_freqs is an array of frequencies
        relative_freqs = [] 
        # Denominator is the number of examples where att == att_val
        denom = len([1 for row in examples if row[att] == att_val])
        if denom == 0:
          continue
        # Loop over all class values
        for class_val in possible_vals['Class']:
          numer = len([1 for row in examples if row[att] == att_val and row['Class'] == class_val])
          relative_freqs.append(numer / denom)
        att_entropy = entropy_calc(relative_freqs, base=len(possible_vals['Class']))
        avg_entropy += att_entropy * (denom / len(examples))  # Weighted average entropy
      # If average entropy is less than previous entropies, set as new lowest entropy
      if avg_entropy < smallest_entropy: 
        smallest_entropy = avg_entropy
        best_att = att
    return best_att, smallest_entropy

  # Recursively builds the decision tree
  def tree_build(examples, node, attributes):
    # check if only row of data, then just add the row class val
    if len(examples) == 1: 
      node.leaf_eval = examples[0]['Class']
      return
    
    # If all examples have the same class, make this a leaf node
    classes = [example['Class'] for example in examples]
    if all(c == classes[0] for c in classes):
      node.leaf_eval = classes[0]
      return

    # If there are no attributes left to split on, make this a leaf node with majority class
    if not attributes:
      temp_lst = [example['Class'] for example in examples]
      node.leaf_eval = max(set(temp_lst), key=temp_lst.count)
      return

    # Find the best attribute to split on
    best_att, smallest_entropy = find_best_attribute(examples, possible_vals, attributes)
    # input(f'best_att = {best_att}, smallest_entropy = {smallest_entropy}')

    if best_att is None:
      temp_lst = [example['Class'] for example in examples]
      node.leaf_eval = max(set(temp_lst), key=temp_lst.count)
      return

    # Set the node's label to the best attribute
    node.label = best_att

    # For each possible value of the best attribute, create child nodes
    for att_val in possible_vals[best_att]:

      # track relative freqs for pruning
      relative_freqs = {} # TODO remove ????
      for thing in [row['Class'] for row in examples]:
        try:
          relative_freqs[thing] += 1
        except KeyError:
          relative_freqs[thing] = 0

      child_node = Node()
      child_node.parent = node
      child_node.relative_freqs = relative_freqs
      node.children[att_val] = child_node

      # Get subset of examples where best_att == att_val
      subset_examples = [example for example in examples if example[best_att] == att_val]

      if not subset_examples:
        # If subset is empty, assign majority class to leaf node
          temp_lst = [example['Class'] for example in examples]
          node.leaf_eval = max(set(temp_lst), key=temp_lst.count)
      else:
        new_attributes = attributes.copy()
        new_attributes.remove(best_att)
        # Recursively build the subtree
        tree_build(subset_examples, child_node, new_attributes)

  # handle missing values, which is denoted with a ?, by replacing it with the row
  temp_examples = []
  for row in examples:
    for attribute, attribute_val in row.items():
      dans_lst = [data[attribute] for data in examples if data[attribute] != '?']
      mode = max(set(dans_lst), key=dans_lst.count)
      if attribute_val == '?': 
        row[attribute] = mode
    temp_examples.append(row)   
  examples = temp_examples

  # Prepare possible values and attributes
  if possible_vals is None:
    possible_vals = {}
    for att in examples[0].keys():
      possible_vals[att] = set()

    for example in examples:
      for att, value in example.items():
        possible_vals[att].add(value)

    for att in possible_vals:
      possible_vals[att] = list(possible_vals[att])

  attributes = list(examples[0].keys())
  attributes.remove('Class')

  # Create the root node and build the tree
  # track relative freqs for pruning
  relative_freqs = {}
  for thing in [row['Class'] for row in examples]:
    try:
      relative_freqs[thing] += 1
    except KeyError:
      relative_freqs[thing] = 0

  root_node = Node()
  root_node.relative_freqs = relative_freqs
  tree_build(examples, root_node, attributes)
  return root_node


def evaluate(node, example):
  '''
  Takes in a tree and one example. Returns the Class value that the tree assigns to the example.
  '''
  
  while len(node.children.keys()) > 0: # if we are not at leaf node
    value = example[node.label] # value of attribute at node
    node = node.children[value] # go to the child at 
  return node.leaf_eval


def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples. Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  temp_examples = []
  for row in examples:
    for attribute, attribute_val in row.items():
      dans_lst = [data[attribute] for data in examples if data[attribute] != '?']
      mode = max(set(dans_lst), key=dans_lst.count)
      if attribute_val == '?': 
        row[attribute] = mode
    temp_examples.append(row)   
  examples = temp_examples

  correct_total = 0
  total = len(examples)
  for row in examples:
    prediction = evaluate(node=node, example=row)
    if prediction == row['Class']:
      correct_total += 1
  return correct_total / total


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples. Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  # implements reduced error pruning
  best_acc = test(node=node, examples=examples)

  def find_leaf_nodes(node):
    leaf_nodes = []
    if len(node.children) == 0: return [node]
    for attribute_val, child_node_obj in node.children.items():
      leaf_nodes.extend(find_leaf_nodes(child_node_obj))
    return leaf_nodes

  improvement = True
  while improvement:
    leaf_nodes = find_leaf_nodes(node)
    improvement = False
    for parent in set([leaf_node.parent for leaf_node in leaf_nodes]):
      if parent.label != node.label:
        temp = parent.children
        parent.children = {}
        # this gets the max class val this node has seen
        parent.leaf_eval = max(parent.relative_freqs, key=parent.relative_freqs.get)
        # test accuracy
        test_acc = test(node=node, examples=examples)
        if test_acc > best_acc: 
          improvement = True
          best_acc = test_acc
        else: parent.children = temp

  return node
  

if __name__ == '__main__':
  from parse import parse

  # train_examples = parse('mushroom.data')
  # examples = parse('tennis.data')
  train_examples = parse('cars_train.data')
  
  default = 0  # NOT USED YET!
  root_node = ID3(examples=train_examples, default=default)

  # Make prediction on a row of data
  row = train_examples[0]
  prediction = evaluate(node=root_node, example=row)
  print(f'Prediction for row: {row} is equal to {prediction}')

  # Test tree on full dataset
  test_examples = parse('cars_test.data')
  accuracy = test(node=root_node, examples=test_examples)
  print(f'Accuracy of tree on test data before pruning = {accuracy:.4f}')

  # Prune tree
  validation_examples = parse('cars_valid.data')
  pruned_tree_root = prune(node=root_node, examples=validation_examples)
  print(f'Tree pruned on validation dataset.')

  # Test again after pruning
  accuracy = test(node=root_node, examples=test_examples)
  print(f'Accuracy of tree on test data after pruning = {accuracy:.4f}')

