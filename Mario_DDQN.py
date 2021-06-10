import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Utils
from tqdm import tqdm
import Gym_Utils
import gym
from gym.wrappers import FrameStack
from gym.spaces import Box
import torch
from torchvision import transforms as T
import torch.optim as optim
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import sys
import gym_super_mario_bros
import gym
import time
import Conv_Models



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

class DDQN_Agent:

    def __init__(self, observation_space, action_space, buffer_size, batch_size):

        self.observation_space = observation_space
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        
        self.gamma = 0.99
        self.tau = 1e-3
        self.learn_step = 0
        self.update = 0 
        
        self.exp_replay = Utils.Expreince_Replay(self.buffer_size)

        self.q_network = Conv_Models.Conv_DDQN(self.observation_space, self.action_space)
        self.q_target = Conv_Models.Conv_DDQN(self.observation_space, self.action_space)
        
        

        self.optimizer = optim.Adam(self.q_network.parameters())
    
    def act(self, state, epsilon):

        if np.random.random_sample() > epsilon:            
            self.q_network.eval()

            with torch.no_grad():
                #state = torch.FloatTensor(state)
                action_values = self.q_network(state)
                
            self.q_network.train()
            
            action = np.argmax(action_values.cpu().data.numpy())
            
            return action
        
        else:
        
            return np.random.randint(env.action_space.n)
    
    
    def learn(self, memory):

        state, action, reward, next_state, done = memory
        self.update += 1

        state = torch.FloatTensor(np.float32(state))

        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state)
        mext_state_q_values = self.q_target(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = mext_state_q_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma + next_q_value * (1-done)
        
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_q_target()

        return loss
    
    def update_q_target(self):
        for source_parameters, target_parameters in zip(self.q_network.parameters(), self.q_target.parameters()):
            target_parameters.data.copy_(self.tau * source_parameters.data + (1.0 - self.tau) * target_parameters.data)
        
        
    def save_weights(self, filename):
        
        torch.save(self.q_network.state_dict(), filename)
    
    def load_weights(self, filename):

        self.q_network.load_state_dict(torch.load(filename))
        self.update_q_target()


Agent = DDQN_Agent(env.observation_space.shape[0],  env.action_space.n, buffer_size = 10_000, batch_size=32)
#Agent.load_weights("Mario_DDQN_1000_2000.pt")
train = True

def run(train, Agent, env):
    all_rewards = []
    reward_windows = deque(maxlen=50)
    losses = []
    loss_window = deque(maxlen=100)
    epsilon_initial = 0.5
    epsilon_decay = 0.999
    epsilon_minium = 0.001
    epsilon = epsilon_initial
    state = env.reset()
    score = 0
    dc = 0
    update_counter = 0
    if train:
        for i in tqdm(range(1, 10_000_00)):
            env.seed(42)
            state = env.reset()
            state = np.array(state)
            action = Agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            Agent.exp_replay.push(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                all_rewards.append(score)
                reward_windows.append(score)
                epsilon = max(epsilon * epsilon_decay, epsilon_minium)
                score = 0

            if len(Agent.exp_replay) > Agent.batch_size:
                mem = Agent.exp_replay.sample(Agent.batch_size)
                loss = Agent.learn(mem)
                
                losses.append(loss.item())
                loss_window.append(loss.item())
                update_counter += 1
            if update_counter % 100 == 0:
                Agent.update_q_target()
            if i % 100 == 0:
                print('\nEpisode:{}/{}|Mean score:{}|Mean Loss:{:.2f}|Dones:{} '.format(i, 1_000, np.mean(reward_windows), np.mean(loss_window), dc), end='')
                sys.stdout.flush()
    else:
        counter = 0
        eps = 0.001
        while counter != 10:
            action = Agent.act(state, eps)
            env.render()
            next_state, r, d, i = env.step(action)
            state = next_state
            if d:
                counter += 1
                state = env.reset()


run(train, Agent, env)

