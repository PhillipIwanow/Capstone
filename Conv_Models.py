import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from Utils import Parametric_Saturated_Relu, Parametric_Saturated_Identity

class Conv_DDQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Conv_DDQN, self).__init__()
        
        self.observation_space=observation_space # with using ~.shape[0] it will giving t=you the input channels
        self.action_space = action_space

        if torch.cuda.is_available():
          self.device = 'cuda'
        else:
          self.device = 'cpu'
        
        
        self.Conv_layer =nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.LeakyReLU()
            )
        
        self.Fully_con_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.action_space)
        )
        

    
    def forward(self, x):
        x = np.array(x)
        x = torch.FloatTensor(x)
        x = x.view(-1, 4, 84, 84)
        x = self.Conv_layer(x)
        x = self.Fully_con_layer(x)

        return x

class C51(nn.Module):
        
    def __init__(self, observation_space, action_space, support, N_atoms=51, device='cpu' ):
        super(C51, self).__init__()
        
        self.device = device
        self.N_atoms = N_atoms
        self.observation_space = observation_space
        self.action_space = action_space
        self.support = support

        self.Layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.action_space *self.N_atoms)

            )
       
    def forward(self, x):

        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q
    
    def dist(self, x):
        x = np.array(x)
        x = torch.FloatTensor(x)
        x = x.view(-1, 4, 84, 84)
        q_atoms = self.Layers(x).view(-1, self.action_space, self.N_atoms)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist





class Conv_PPO(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_size=512, n_layers=2, std=0.0):
        super(Conv_PPO, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
    
        self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.Flatten()
 
        )

        
   
        
        self.Critic = nn.Sequential(
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

        self.Actor = nn.Sequential(
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_space),
            nn.Softmax(dim=-1)      
        )

    

  

       
    def forward(self, x):
        x = np.array(x)
        x = torch.FloatTensor(x)
        x = x.view(-1, 4, 84, 84)
        x = self.conv_layer(x)
        value = self.Critic(x)
        dist = self.Actor(x)
        dist = Categorical(dist)
        

        return dist, value






class Conv_NMD(nn.Module):
        def __init__(self, observation_space, action_space, hidden_size=512, n_layers=2, std=0.0):
            super(Conv_NMD, self).__init__()


            self.psr = Parametric_Saturated_Relu()
            self.par_id = Parametric_Saturated_Identity()

            self.m = torch.ones(4)

            self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.Flatten())

            #Feedfoward -> "Actor/critic": begin
            self.Fc1 = nn.Linear(4096, 512)
            self.Fc2 = nn.Linear(512, 256)
            self.Fc3 = nn.Linear(256, 256)
            self.actor_out = nn.Linear(256, action_space)
            self.critic_out = nn.Linear(256, 1)
            #Feedfoward : End

            self.Neuromod = nn.Sequential(
                nn.GRU(4098, 512),
                nn.ReLU(),
                nn.GRU(512, 256),
                nn.ReLU(),
                nn.GRU(256, 4)
            )
            

