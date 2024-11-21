import torch
import torch.nn as nn
import torch.nn.functional as F

# set random seed for reproducibility
torch.manual_seed(343)

# define network for Question 1
class ANN(nn.Module):
    def __init__(self,k):
        #initialize parent class nn.Module 
        super(ANN, self).__init__()
        #first linear layer with 3 input features, 'k' output neurons (no bias as it is appended separately)
        self.layer1 = nn.Linear(3, k, bias=False) # bias is appended to all samples from read_data function
        #second linear layer with 'k' input nerons, 2 output nerons (no bias as it is appended separately)
        self.layer2 = nn.Linear(k, 2, bias=False)

    def forward(self, input_vals):
        #convert input values to torch tensor if not already
        input_vals = torch.Tensor(input_vals)
        #first layer activation function with ReLU (rrectified linear unit) function
        activation_1 = F.relu(self.layer1(input_vals)) # chad relu
        #output layer without activation (for compatibility with loss function)
        output = self.layer2(activation_1) 
        return output

def train(self, lr: float, num_epochs: int):
    """Trains network and returns train and validation losses"""
    self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, **optimizer_kwargs)
    train_losses, valid_losses = [], []
    for epoch in range(1, num_epochs+1):
        #perform 1 full pass over training data; get avg loss for epoch, is use_regularizer is True, use the regularized training fucntion
        train_loss = self._train_epoch()
        #preform 1 full pass over validation data; get avg loss for epoch
        valid_loss = self._valid_epoch()
        #append training and validation for currrent epoch to their list
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    #return lists of training and validation losses from all epochs
    return train_losses, valid_losses