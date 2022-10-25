import torch
import torch.nn as nn
import torch.nn.functional as F

class Duel_QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units_1= 64, hidden_units_2=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Duel_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.A_fc1 = nn.Linear(state_size, hidden_units_1)
        self.A_fc2 = nn.Linear(hidden_units_1, hidden_units_2)
        self.A_fc3 = nn.Linear(hidden_units_2, action_size)
        
        self.V_fc1 = nn.Linear(state_size, hidden_units_1)
        self.V_fc2 = nn.Linear(hidden_units_1, hidden_units_2)
        self.V_fc3 = nn.Linear(hidden_units_2, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        a = F.relu(self.A_fc1(state))
        a = F.relu(self.A_fc2(a))
        a = self.A_fc3(a)
        
        v = F.relu(self.V_fc1(state))
        v = F.relu(self.V_fc2(v))
        v = self.V_fc3(v)
        output = v  + a -  a.mean(dim= 1, keepdim=True)
        
        return output
