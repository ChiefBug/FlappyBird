"""
AUTHOR:         Dhyey Shingala, Aviral Shrivastava, Elham Inamdar
FILENAME:       dqn.py
SPECIFICATION:  This file contains the DQN class that have the various parameters set for the DQN algorithm.
FOR:            CS 5392 Reinforcement Learning Section 001
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    """
    NAME: DQN
    PURPOSE: The DQN Class contains the for using DQN algorithm.
    INVARIANTS: The class should have a convolutional neural network consisting of 3 convolutional layers and 2 fully
    connected layers. The input state should be a tensor of shape (batch_size, 1, height, width) and the output should
    be a tensor of shape (batch_size, num_actions). The class should have an optimizer for the neural network parameters
    and a loss function for training the network.
    """
    def __init__(self, lr):
        """
        NAME: lr
        PARAMETERS: None
        PURPOSE: Define convolutional and linear layers for the DQN model
        PRECONDITION: DQN object has been initialized
        POSTCONDITION: Convolutional and linear layers have been defined and stored in the DQN object
        """
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3840,512)
        self.fc2 = nn.Linear(512, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        """
        NAME: forward
        PARAMETERS: state: tensor representing the state of the environment
        PURPOSE: Compute the forward pass of the neural network given the state of the environment
        PRECONDITION:
        - 'state' must be a tensor with dimensions (batch_size, channels, height, width)
        - All layers in the network must be properly initialized
        POSTCONDITION: Return a tensor with dimensions (batch_size, num_actions), representing the estimated Q-values for each action
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3840)
        x = F.relu(self.fc1(x))
        return self.fc2(x)