import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys 
import pandas as pd

class Expreince_Replay(object):

    def __init__(self, capacity):
        
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):

        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):

        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        
        return len(self.buffer)

class policy_memory:
    def __init__(self, batch_size):
        
        self.batch_size = batch_size

        self.log_probs  = []
        self.actions = []
        self.values = []
        self.states = []
        self.rewards = []
        self.dones = []
    
    def push(self, state, prob, value, action, reward, done):
        self.log_probs.append(prob)
        self.values.append(value)
        self.states.append(np.array(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def gen_batches(self):

       n_states = len(self.states)

       batch_start = np.arange(0, n_states, self.batch_size)
       indices = np.arange(n_states, dtype=np.int64)
       np.random.shuffle(indices)
       batches = [indices[i:i+self.batch_size] for i in batch_start]

       return np.array(self.states), np.array(self.actions), np.array(self.log_probs), np.array(self.values), np.array(self.rewards), np.array(self.dones), batches

    def clear_buffer(self):
        del self.log_probs[:]
        del self.values[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]




class Parametric_Saturated_Relu(nn.Module):
    def __init__(self):
        super(Parametric_Saturated_Relu, self).__init__()
        self.a = nn.Parameter(torch.rand(1))
    
    def forward(self, x, m):

        output = torch.selu(m * x + self.a)

        return output

class Parametric_Saturated_Identity(nn.Module):
    def __init__(self):
        super(Parametric_Saturated_Identity, self).__init__()
        self.a = nn.Parameter(torch.rand(1))
    
    def forward(self, x, m):

        output = m * x + self.a

        return output

