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
        self.N_atoms = N_atoms # standard is set to 51 as dicussed in the paper, this will be how many distrubtions you will have so in case of 51 you will have distrubition of 51
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
            nn.Linear(512, 1)
        )

        self.Actor = nn.Sequential(
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_space),
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
            self.prev_state = 0
            self.m = torch.ones(4)
            
            self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.Flatten())

            self.modulator = nn.Sequential(
            nn.Linear(4098, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4)
        )


            #Feedfoward -> "Actor/critic": begin
            self.Fc1 = nn.Linear(4096, 512)
            self.Fc2 = nn.Linear(512, 256)
            self.Fc3 = nn.Linear(256, 256)
            self.actor_out = nn.Linear(256, action_space)
            self.critic_out = nn.Linear(256, 1)
            #Feedfoward : End

         
            
        def forward(self, x):
            
            x = np.array(x)
            x = torch.FloatTensor(x)
            x = x.view(-1, 4, 84, 84)
            x = self.conv_layer(x)
        

            x = self.Fc1(x)
            x = self.psr(x, self.m[0])
            x = self.Fc2(x)
            x = self.psr(x, self.m[1])
            x = self.Fc3(x)
            x = self.psr(x, self.m[2])
            
            
            
            actor = self.actor_out(x)
            
        
            actor = F.softmax(actor, dim=-1)
            actor = actor.clamp(min=1e-3)

            
            
            critic = self.critic_out(x)
            critic = self.par_id(critic, self.m[3])
            value = critic
            try:
                dist = Categorical(actor)
            except:
                print(actor)

            return dist, value
        
        def modulate(self, last_state, last_action, last_reward):
            last_state = np.array(last_state)
            last_state = torch.FloatTensor(last_state)
            last_state = last_state.view(-1, 4, 84, 84)
            last_state = self.conv_layer(last_state)
            mod_in = torch.tensor([last_action, last_reward]).view(1, 2)
            mod_in = torch.cat((last_state, mod_in), 1).float()
            m = self.modulator(mod_in)
            self.m = m.view(4)

        
