"""
AUTHOR:         Dhyey Shingala, Aviral Shrivastava, Elham Inamdar
FILENAME:       agent.py
SPECIFICATION:  This file contains the implementation of the Agent class for reinforcement learning using DQN algorithm.
FOR:            CS 5392 Reinforcement Learning Section 001
"""

import numpy as np
import torch as T
from dqn import DQN
import random
from memory import Memory

class Agent(object):
    """
    NAME: Agent
    PURPOSE: The Agent Class contains the implementation of the Agent for reinforcement learning using DQN algorithm.
    INVARIANTS: The Agent has a memory of previous experiences, a neural network to approximate the Q-function,
    and methods for selecting actions, saving and loading experiences, replacing the target network, and updating the Q-function.
    """
    def __init__(self, gamma, epsilon, lr, n_actions, mem_size,
                    batch_size, eps_min=0.01, eps_dec=5e-7, replace=1000):
        """
        NAME: __init__
        PURPOSE: Initialize the Agent object with given parameters and create instances of DQN and Memory
        PARAMETERS:
        gamma: float, discount factor for future rewards
        epsilon: float, initial exploration probability
        lr: float, learning rate
        n_actions: int, number of possible actions
        mem_size: int, size of the replay buffer memory
        batch_size: int, number of samples to take from the replay buffer memory for training the DQN
        eps_min: float, minimum exploration probability
        eps_dec: float, rate at which exploration probability decreases
        replace: int, number of steps before updating the target DQN
        PRECONDITION: None
        POSTCONDITION: The Agent object is initialized with the given parameters and the DQN neural networks are created.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnr = replace
        self.learn_step_counter = 0

        self.memory = Memory(mem_size, n_actions)

        self.q_eval = DQN(self.lr)
        self.q_next = DQN(self.lr)

    def action(self, state):
        """
        NAME: action
        PARAMETERS: state: numpy.ndarray
        PURPOSE: Selects an action based on the current state and epsilon - greedy policy.
        PRECONDITION: The Agent object must be initialized and have a valid neural network.
        POSTCONDITION: The action is selected and returned along with its index.
        """
        if np.random.random() > self.epsilon:
            actions = self.q_eval.forward(state)
            action_pure = T.argmax(actions).item()
            action = [1,0] if action_pure == 0 else [0,1]
        else:
            action = random.choice([[1,0],[0,1]])
            action_pure = 0 if action == [1,0] else 1
        
        return action, action_pure
    
    def save(self, state, action, reward, state_, done):
        """
        NAME: save
        PARAMETERS: state: numpy.ndarray, action: int, reward: float, state_: numpy.ndarray, done: bool
        PURPOSE: Saves the experience tuple to the replay memory.
        PRECONDITION: The Agent object must be initialized and have a valid memory object.
        POSTCONDITION: The experience tuple is saved to the memory object.
        """
        self.memory.save(state, action, reward, state_, done)

    def load(self):
        """
        NAME: load
        PARAMETERS: None
        PURPOSE: Loads the experience tuples from the replay memory.
        PRECONDITION: The Agent object must be initialized and have a valid memory object.
        POSTCONDITION: The experience tuples are loaded from the memory object.
        """
        state, action, reward, new_state, done = self.memory.load(self.batch_size)
        return state, action, reward, new_state, done

    def replace_target_network(self):
        """
        NAME: replace_target_network
        PARAMETERS: None
        PURPOSE: Replace target network weights with evaluation network weights periodically
        PRECONDITION: Evaluation network and target network have the same architecture
        POSTCONDITION: Target network weights are updated to evaluation network weights every self.replace_target_cnr steps
        """
        if self.learn_step_counter % self.replace_target_cnr == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        """
        NAME: decrement_epsilon
        PARAMETERS: None
        PURPOSE: Decrement exploration probability (self.epsilon) linearly
        PRECONDITION: None
        POSTCONDITION: self.epsilon is decremented by self.eps_dec or set to self.eps_min if it is below self.eps_min
        """
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        """
        NAME: learn
        PARAMETERS: None
        PURPOSE: Perform one step of training for the Q network using a random batch of experiences from memory
        PRECONDITION: Memory contains at least self.batch_size amount of data
        POSTCONDITION: Updates the weights of the evaluation network using the DQN loss with respect to the target network
        """
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.load()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()