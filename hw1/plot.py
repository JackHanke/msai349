import matplotlib.pyplot as pyplot
from ID3 import *
from parse import parse
import numpy as np

# collect data
raw_examples = parse('cars_train.data')
n = len(raw_examples)
thing = [i for i in range(n)]

training_attempts = []
unpruned_accs = []
pruned_accs = []
for _ in range(100):
    for training_size in range(10,300,10):
        training_attempts.append(training_size)

        indexes = np.random.choice(thing, size=training_size)
        train_examples, test_examples = [], []
        for index, row in enumerate(raw_examples):
            if index in indexes: train_examples.append(row)
            else: test_examples.append(row)

        root_node = ID3(examples=train_examples, default=0)
        unpruned_acc = test(node=root_node, examples=test_examples)
        unpruned_accs.append(unpruned_acc)

        # pruned_acc = 
        # pruned_accs.append(pruned_acc)

plt.scatter(x=training_attempts, y=unpruned_accs)
plt.scatter(x=training_attempts, y=pruned_accs)
plt.show()
