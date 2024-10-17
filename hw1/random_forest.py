from ID3 import ID3, evaluate
import random
from node import Node


def random_forest(examples: list[dict[str, any]], n_trees: int = 3, possible_vals: dict[str, set] = None, feature_subset_size: int = 2) -> list[Node]:
    forest = []

    for _ in range(n_trees):
        # Get subset using bootstrap sampling
        example_subset = [random.choice(examples) for _ in range(len(examples))]
        
        # Get attributes to randomly sample a subset
        attributes = list(examples[0].keys())
        attributes.remove('Class')

        # Make feature subset
        feature_subset = list(set([random.choice(attributes) for _ in range(feature_subset_size)])) + ['Class']

        # Get final subset
        subset = [{key: example[key] for key in feature_subset if key in feature_subset} for example in example_subset]

        # Run ID3
        tree = ID3(examples=subset, default=0, possible_vals=possible_vals)

        # Append tree to forest
        forest.append(tree)
    return forest


def evalulate_random_forest(forest: list[Node], example: dict[str, any]) -> any:
    preds = [evaluate(node=node, example=example) for node in forest] # Get preds for each node
    return max(set(preds), key=preds.count) # Return mode


def test_random_forest(forest: list[Node], examples: list[dict[str, any]]) -> float:
    total_preds = 0
    total_correct = 0
    for example in examples:
        # Get true 
        y_true = evalulate_random_forest(forest=forest, example=example)
        # Get pred
        y_pred = example['Class']
        # Increment true 
        total_preds += 1
        if y_pred == y_true:
            # Increcment correct
            total_correct += 1
        # Return accruacy
        return float(total_correct/total_correct)
    
if __name__ == '__main__':
  from parse import parse

  # train_examples = parse('mushroom.data')
  # examples = parse('tennis.data')
  train_examples = parse('cars_train.data')
  
  model = random_forest(examples=train_examples)

  # Make prediction on a row of data
  row = train_examples[0]
  prediction = evalulate_random_forest(forest=model, example=row)
  print(f'Prediction for row: {row} is equal to {prediction}')

  # Prune tree
  validation_examples = parse('cars_valid.data')

  # Test again after pruning
  accuracy = test_random_forest(forest=model, examples=validation_examples)
  print(f'Accuracy of tree on validation examples = {accuracy:.4f}')