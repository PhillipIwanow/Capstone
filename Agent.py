import gym 
import math
import random
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from collections import namedtuple, deque
import utils
from DQN import Dqn_network
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Agent():

    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)

        #q networks 
        self.qnet = Dqn_network(state_size, action_size).to(device)
        self.qnet_fixed = Dqn_network(state_size, action_size).to(device)
        self.batch_size = self.qnet.batch_size
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=1e-4)
        
        self.memory = utils.ReplayMemory( 64, 100000, self.seed)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1 #TODO add update_every global var

        if self.t_step % 2 == 0:
            
            if len(self.memory) >= self.batch_size:
               
                experience = self.memory.sample()
                self.learn(experience)
    
    def act(self, state, epsilon):
        
        rand_choice = random.random()
        if rand_choice < epsilon:
            
            return np.random.randint(self.action_size), 'r'
        else:
            
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnet.eval()
            with torch.no_grad():
                action_values = self.qnet(state)
            self.qnet.train()
            return np.argmax(action_values.cpu().data.numpy()), 'c'

    def learn(self, experience):
            

            state, actions, rewards, next_state, dones = experience          
            criterion = torch.nn.MSELoss()
            action_vals = self.qnet_fixed(next_state).detach()
            max_action_values = action_vals.max(1)[0].unsqueeze(1)
            
            Q_target = rewards + (self.qnet.gamma * max_action_values * (1 - dones))
            
            Q_expected = self.qnet(state).gather(1, actions)
            
            loss = criterion(Q_expected, Q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.soft_update(self.qnet, self.qnet_fixed) #TODO add TAU global variable 

    def soft_update(self, qnet, fixed, TAU=0.001):

        for param, target_param in zip(qnet.parameters(), fixed.parameters()):
            target_param.data.copy_(TAU*param.data + (1.0-TAU)*target_param.data)

        

                


env = gym.make('LunarLander-v2')
input_shape = env.observation_space
vars = Dqn_network(1,1)

Agent = Agent(8, 4, 0)

train = True


def run(n_episodes=vars.number_of_iterations, max_t = 1000, eps_initial=vars.initial_epsilon, eps_final=vars.final_epsilon,
        eps_decay=0.9996):
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_initial
    print(eps)
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action, choice = Agent.act(state, eps)
            
            next_state, reward, done, _ = env.step(action)
            Agent.step(state, action, reward, next_state, done)
            state = next_state
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps*eps_decay, eps_final)
        print('\rEpisode{}\tAverge Score {:.2f}\tEpsilon{:.2f}\tChoice {}\tAction{}'.format(i_episode, np.mean(scores_window), eps, choice, action), end='')
        
        if i_episode % 500 == 0:
            #torch.save(Agent.qnet, 'DQN{}.pt'.format(i_episode))
            pass

        if i_episode %100==0:
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
            
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                        np.mean(scores_window)))

            torch.save(Agent.qnet, 'DQN_solved.pt')
            break
    return scores

if train:

    scores = run()
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.plot(pd.Series(scores).rolling(100).mean())

    plt.title('DQN TRAINING')
    plt.xlabel('# of episodes')
    plt.ylabel('scores')
    plt.show()

else:
    Agent.qnet = torch.load('DQN_solved.pt')
    for i in range(10):
        state = env.reset()
        score = 0
        while True:
            env.render()
            action, _ = Agent.act(state, 0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break
        print('episode: {} score: {:.2f}'.format(i, score))
    env.close()