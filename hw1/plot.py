import matplotlib.pyplot as plt
from ID3 import *
from parse import parse
import numpy as np

# collect data
raw_examples = parse('house_votes_84.data')
possible_vals = {}
for att in raw_examples[0].keys():
    possible_vals[att] = set()
for example in raw_examples:
    for att, value in example.items():
        possible_vals[att].add(value)
for att in possible_vals:
    possible_vals[att] = list(possible_vals[att])

n = len(raw_examples)
thing = [i for i in range(n)]

training_attempts = []
unpruned_accs = []
pruned_accs = []
attempts = 5
for training_size in range(10,300,10):
    training_attempts.append(training_size)
    unpruned_accs.append(0)
    pruned_accs.append(0)
    for _ in range(attempts):
        unpruned_acc = 0
        indexes = np.random.choice(thing, size=training_size//2)
        validation_indexes = np.random.choice(thing, size=training_size//2)
        train_examples, validation_examples, test_examples = [], [], []
        for index, row in enumerate(raw_examples):
            if index in indexes: train_examples.append(row)
            elif index in validation_indexes: validation_examples.append(row)
            else: test_examples.append(row)

        root_node = ID3(examples=train_examples, default=0, possible_vals=possible_vals)
        unpruned_acc = test(node=root_node, examples=test_examples)
        unpruned_accs[-1] += ((unpruned_acc)/attempts)
        print(f'>> An unpruned accuracy for {training_size} = {unpruned_acc}')

        pruned_node = prune(node=root_node, examples=validation_examples) # TODO
        pruned_acc = test(node=pruned_node, examples=test_examples)
        pruned_accs[-1] += ((pruned_acc)/attempts)
        print(f'>> A pruned accuracy for {training_size} = {unpruned_acc}')

    print(f'Completed training size {training_size}')

plt.scatter(x=training_attempts, y=unpruned_accs)
plt.scatter(x=training_attempts, y=pruned_accs)
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title("Training size for US House 1984 Dataset")
plt.show()
