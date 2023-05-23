"""
AUTHOR:         Dhyey Shingala, Aviral Shrivastava, Elham Inamdar
FILENAME:       memory.py
SPECIFICATION:  This file has the memory class saves and loads the reward points from memory.
FOR:            CS 5392 Reinforcement Learning Section 001
"""
import numpy as np
import torch

class Memory(object):
    """
    NAME: Memory
    PURPOSE: This class saves and loads the points of the run.
    INVARIANTS: The max_size parameter in the constructor must be a positive integer. The n_actions parameter in the
    constructor must be a positive integer.
    """
    def __init__(self, max_size, n_actions):
        """
        NAME: init
        PARAMETERS: max_size, a positive integer representing the maximum size of the memory
        n_actions, a positive integer representing the number of possible actions
        PURPOSE: This method initializes the attributes of the Memory object
        PRECONDITION: The max_size and n_actions parameters must be positive integers
        POSTCONDITION: The mem_size, mem_cntr, state_memory, new_state_memory, action_memory, reward_memory, and terminal_memory attributes are initialized with the correct dimensions and data types
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, 1, 80, 112))
        self.new_state_memory = torch.zeros((self.mem_size, 1, 80, 112))
        self.action_memory = torch.zeros((self.mem_size), dtype=torch.long)
        self.reward_memory = torch.zeros(self.mem_size)
        self.terminal_memory = torch.zeros((self.mem_size), dtype=torch.bool)

    def save(self, state, action, reward, state_, done):
        """
        NAME: save
        PARAMETERS: state, a tensor representing the current state
        action, a tensor representing the current action
        reward, a float representing the current reward
        state_, a tensor representing the next state
        done, a bool representing whether the episode is finished or not
        PURPOSE: This method saves the current state, action, reward, next state, and done values into the memory buffer
        PRECONDITION: The Memory object must have been initialized, and the parameters must be of the correct data type and dimensions
        POSTCONDITION: The current state, action, reward, next state, and done values are stored in the memory buffer at the correct index, and the mem_cntr attribute is incremented
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def load(self, batch_size):
        """
        NAME: load
        PARAMETERS: batch_size, an integer specifying the number of samples to load
        PURPOSE: This function randomly selects a batch of samples from memory and returns them
        PRECONDITION: Memory must contain at least `batch_size` samples
        POSTCONDITION: Returns five numpy arrays containing `batch_size` samples of the states, actions, rewards,
                       new states, and terminal states respectively.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal