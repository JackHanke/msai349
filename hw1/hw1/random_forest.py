from ID3 import ID3, evaluate
import random
from node import Node
import argparse
from parse import parse

def random_forest(examples: list[dict[str, any]], n_trees, feature_subset_size, possible_vals: dict[str, set] = None) -> list[Node]:
    forest = []

    for _ in range(n_trees):
        # Get subset using bootstrap sampling
        example_subset = [random.choice(examples) for _ in range(len(examples))]
        
        # Get attributes to randomly sample a subset
        attributes = list(examples[0].keys())
        attributes.remove('Class')

        # Make feature subset
        assert feature_subset_size <= len(attributes)
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
    temp_examples = []
    for row in examples:
        for attribute, attribute_val in row.items():
            dans_lst = [data[attribute] for data in examples if data[attribute] != '?']
            mode = max(set(dans_lst), key=dans_lst.count)
            if attribute_val == '?': 
                row[attribute] = mode
        temp_examples.append(row)   
    examples = temp_examples
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
    return total_correct/total_preds
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Random Forest with ID3 decision trees')
    parser.add_argument('--train_path', type=str, default='tennis.data', help='Path to the training dataset')
    parser.add_argument('--val_path', type=str, default='tennis.data', help='Path to the validation dataset')
    parser.add_argument('--n_trees', type=int, default=2, help='Number of trees in the random forest')
    parser.add_argument('--feature_subset_size', type=int, default=2, help='Size of the feature subset for each tree')

    args = parser.parse_args()

    # Load training and validation data
    train_examples = parse(args.train_path)
    val_examples = parse(args.val_path)

    # Train random forest
    print('Fitting random forest on {}'.format(args.train_path))
    forest = random_forest(examples=train_examples, n_trees=args.n_trees, feature_subset_size=args.feature_subset_size)

    # Evaluate on validation data
    accuracy = test_random_forest(forest=forest, examples=val_examples)
    
    print(f'Accuracy of the random forest on {args.val_path} = {accuracy:.4f}')