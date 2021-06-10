import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys 
import pandas as pd

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


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

class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

def plot(rewards, losses):

    rewards = np.array(rewards)
    losses = np.array(losses)

    plt.figure(figsize=(64, 64))
    plt.subplot(131)
    
    plt.title('Rewards')
    plt.plot(rewards, color='blue')
    plt.plot(pd.Series(rewards).rolling(50).mean(), color='red')

    plt.subplot(132)
    
    plt.title('loss')
    plt.plot(losses, color='blue')
    plt.plot(pd.Series(losses).rolling(100).mean(), color='red')

    plt.show()


def run(env, Agent, n_frames=20_000, train=True, savefile=None, loadfile=None):
    
    epsilon_initial = 1.0
    epsilon_decay = 0.99
    epsilon_minium = 0.01
    epsilon = epsilon_initial

    all_rewards = []
    score_window = deque(maxlen=25)
    episode_losses = []
    losses = []
    loss_window = deque(maxlen=100)
    state = env.reset()
    score = 0
    dc = 0
    run_til_solved = False
    if train:
        while run_til_solved !=True:

            action = Agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            loss = Agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if loss != 0:
                losses.append(loss)
            
            if done:
                state = env.reset()
                all_rewards.append(score)
                score_window.append(score)
                score = 0
                dc += 1
                
            epsilon = max(epsilon * epsilon_decay, epsilon_minium)
              
            
            print('\r|Mean Rewards:{}|Mean Loss{}|dones:{}'
                .format(np.mean(score_window), np.mean(losses), dc), end='')
            sys.stdout.flush()

            if np.mean(score_window) >= 200:
                run_til_solved = True

        return all_rewards, losses