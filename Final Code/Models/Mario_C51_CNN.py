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


env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v1')
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

class c51_Agent:

    def __init__(self, observation_space, action_space, buffer_size, batch_size, target_update, v_min=0.0, v_max=200.0, n_atoms=51):

        self.observation_space = observation_space
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms


        self.gamma = 0.99
        self.tau = 1e-3
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms)

        self.exp_replay = Utils.Expreince_Replay(self.buffer_size)


        self.q_network = Conv_Models.C51(self.observation_space, self.action_space, self.support)
        self.target_network = Conv_Models.C51(self.observation_space, self.action_space, self.support)
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters())
    
    def act(self, state, epsilon):

        if epsilon > np.random.random():
            action = np.random.randint(self.action_space)
        
        else:
            action = self.q_network(state)
            action = action.detach().numpy()
            action = action.argmax()

        return action
    
    def step(self, state, action, reward, next_state, done):
        
        self.exp_replay.push(state, action, reward, next_state, done)
    
    def update_model(self):

        sample = self.exp_replay.sample(self.batch_size)

        loss = self.compute_loss(sample)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_q_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def learn(self, sample):

        state, action, reward, next_state, done = sample

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward).view(-1, 1)
        done = torch.FloatTensor(done).view(-1, 1)

        delta_z = float(self.v_max - self.v_min) / (self.n_atoms - 1)

        with torch.no_grad():
            next_action = self.target_network(next_state).argmax(1)
            next_dist = self.target_network.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()


            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.n_atoms, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.n_atoms)
                
            )

            proj_dist = torch.zeros(next_dist.size())
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.q_network.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])

        loss = -(proj_dist * log_p).sum(1).mean()

        return loss
    
    def save_weights(self, filename):
        
        torch.save(self.q_network.state_dict(), filename)
    
    def load_weights(self, filename):

        self.q_network.load_state_dict(torch.load(filename))
        self.update_q_target()

Agent = c51_Agent(env.observation_space.shape[0], env.action_space.n, buffer_size=50_000, batch_size=32, target_update=100)

train = True

def run(train, Agent, env):
    all_rewards = []
    reward_windows = deque(maxlen=100)
    losses = []
    loss_window = deque(maxlen=100)
    epsilon_initial = 1.0
    epsilon_decay = 0.995
    epsilon_minium = 0.1
    epsilon = epsilon_initial
    state = env.reset()
    score = 0
    dc = 0
    update_counter = 0
    if train:
        for i in tqdm(range(1, 100_001)):
            state = np.array(state)
            epsilon = max(epsilon * epsilon_decay, epsilon_minium)
            action = Agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            Agent.exp_replay.push(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                all_rewards.append(score)
                reward_windows.append(score)
                state = env.reset()
                score = 0

            if len(Agent.exp_replay) > Agent.batch_size:
                mem = Agent.exp_replay.sample(Agent.batch_size)
                loss = Agent.learn(mem)
                
                losses.append(loss.item())
                loss_window.append(loss.item())
                update_counter += 1
            if update_counter % 1000 == 0:
                Agent.update_q_target()
            if i % 1000 == 0:
                print('Episode:{}/{}|Mean score:{}|Mean Loss:{:.2f}|Dones:{} '.format(i, 1_000, np.mean(reward_windows), np.mean(loss_window), dc))
        return all_rewards, losses
    else:
        counter = 0
        eps = 0.001
        score = 0
        while counter != 10:
            action = Agent.act(state, eps)
            env.render()
            next_state, r, d, i = env.step(action)
            state = next_state
            score += r
            if d:
                print(score)
                counter += 1
                state = env.reset()
                score = 0

Agent.load_weights("mario_c51_150k.pt")
all_rewards, losses = run(train, Agent, env)
Agent.save_weights("mario_c51_350k.pt")

rewards_filename = '_c51_Rewards_3500k.npy'
losses_filename = 'c51_Losses_350k.npy'

with open(rewards_filename, 'wb') as f:
  np.save(f, all_rewards)

with open(losses_filename, 'wb') as f:
  np.save(f, losses) 