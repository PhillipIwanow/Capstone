import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DDQN(nn.Module):
    
    def __init__(self, observation_space, action_space, device):
        super(DDQN, self).__init__()
        
        self.device = device
        
        self.Network = nn.Sequential(
            
            nn.Linear(observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
            
              
        )
    
    def forward(self, x):

        return self.Network(x)



class C51(nn.Module):
        
    def __init__(self, observation_space, action_space, support, N_atoms=51, device='cpu' ):
        super(C51, self).__init__()
        
        self.device = device
        self.N_atoms = N_atoms
        self.observation_space = observation_space
        self.action_space = action_space
        self.support = support

        self.Layers  = nn.Sequential(
            nn.Linear(self.observation_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space * self.N_atoms)
        )
    
    def forward(self, x):

        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q
    
    def dist(self, x):

        q_atoms = self.Layers(x).view(-1, self.action_space, self.N_atoms)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist
        
    

class policy(nn.Module):
    
    def __init__(self, observation_space, action_space, device):
        super(policy, self).__init__()

        self.device = device

        self.network = nn.Sequential(
            nn.Linear(observation_space, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, action_space),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        
        action_proba = self.network(torch.FloatTensor(x))
        return action_proba



class Actor(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Actor, self).__init__()
        
        self.input_layer = nn.Linear(observation_space, 64)
        self.hidden_1 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, action_space)
    
    def forward(self, x, z):
        x = self.input_layer(x)
        x = F.tanh(x * (z[0] * 1))
        x = self.hidden_1(x)
        x = F.tanh(x * (z[1] * 1))
        x = self.output_layer(x)
        x = F.softmax(x, dim=-1)

        return x

class Critic(nn.Module):
    def __init__(self, observation_space):
        super(Critic, self).__init__()

        self.input_layer = nn.Linear(observation_space, 64)
        self.hidden1 = nn.Linear(64, 32)
        self.modout = nn.Linear(32, 2)
        self.output_layer = nn.Linear(32, 1)
    
    def forward(self, x):

        x = self.input_layer(x)
        x = F.tanh(x)
        x = self.hidden1(x)
        x = F.tanh(x)
        mod = self.modout(x)
        mod = 1 + F.tanh(x)
        mod.clamp(min=0.001)
        x = self.output_layer(x)
        x = F.softmax(x, dim=-1)

        return x, mod
