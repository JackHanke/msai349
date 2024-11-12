
# define network for Question 1
class Net(nn.Module):
    def __init__(self,k):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(1000, 100, bias=True) # bias is appended to all samples from read_data function
        self.layer2 = nn.Linear(100, 29, bias=True)

    def forward(self, input_vals):
        input_vals = torch.Tensor(input_vals)
        activation_1 = F.relu(self.layer1(input_vals)) # chad relu
        activation_2 = F.softmax(self.layer2(activation_1)) # add argument for dim
        return activation_2


if __name__ == '__main__':
    train_cost_history, valid_cost_history = [], []
    # bring in training data
    train_data, train_data_ys, train_data_onehot = read_data(name=dataset, mode='train')
    # instantiate network for specific k_val
    net = Q1_Net(k=k_val)
    # create cost function 
    cost_function = nn.CrossEntropyLoss()
    # create stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # set network to train mode because PyTorch docs say so
    net.train()
    # for each epoch
    for epoch in range(1,num_epochs+1):
        # implemenst batch processing
        batched_train_data = batchify(data=train_data, batch_size=batch_size)
        batched_train_data_onehot = batchify(data=train_data_onehot, batch_size=batch_size)
        for batched_data, batched_data_onehot in zip(batched_train_data, batched_train_data_onehot):
            # forward pass
            inferences = net(input_vals=batched_data)
            # computes cost of forward pass
            cost = cost_function(inferences, batched_data_onehot)
            # calculates gradient
            cost.backward()
            # update weights and biases of network
            optimizer.step()
            # zero gradients bc PyTorch docs said so
            optimizer.zero_grad()
        # record training cost per epoch for plotting
        train_cost_history.append(float(cost))

        # get validation data
        valid_data, valid_data_ys, valid_data_onehot = read_data(name=dataset, mode='valid')
        # conduct validation inferences
        validation_inferences = net(input_vals=valid_data)
        # compute validation cost
        cost = cost_function(validation_inferences, valid_data_onehot)
        # record validation cost per epoch for plotting
        valid_cost_history.append(float(cost))
        
    plt.plot([i for i in range(1, num_epochs+1)], train_cost_history, label=f'Train k={k_val}')
    plt.plot([i for i in range(1, num_epochs+1)], valid_cost_history, label=f'Valid k={k_val}')

    plt.title(f'Cost over epochs of NN with k hidden neurons on {dataset} dataset')
    plt.xlabel(f'Number of Epochs')
    plt.ylabel(f'Cross Entropy Loss')
    plt.legend(loc='upper right')
    plt.show()