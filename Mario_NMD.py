from numpy.core.fromnumeric import clip
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
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)


class ppo:

    def __init__(self, observation_space, action_space):

        self.observation_space = observation_space
        self.action_space = action_space

        self.gamma = 0.99
        self.tau = 0.95
        self.epochs = 4
        self.clip = 0.2
        self.ppo_epochs = 5

        self.memory = Utils.policy_memory(batch_size=1)
        
        
        self.NMD = Conv_Models.Conv_NMD(self.observation_space, self.action_space)
        self.NMD_opt = optim.Adam(self.NMD.parameters())
        
        
    def get_model_out(self, state, reward):
        
        dist, value = self.NMD(state, reward)
        return dist, value

      

 
    def update(self):
        for _ in range(self.epochs):
            state_T, action_T, old_prob_T, values_T, rewards_T, dones_T, batches = self.memory.gen_batches()

            values = values_T
            advantage = torch.zeros(len(rewards_T))

            for t in range(len(rewards_T) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_T) - 1):
                    a_t += discount*(rewards_T[k] + self.gamma*values[k+1]*(1-dones_T[k]) - values[k])
                    discount *= self.gamma*self.tau
                advantage[t] = a_t
            
            for batch in batches:
                

                states = state_T[batch]
                old_probs = old_prob_T[batch]
                actions = action_T[batch]
                rewards = rewards_T[batch]

                dist, value = self.get_model_out(states, rewards)

                new_probs = dist.log_prob(actions)
               

                ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * ratio
                weighted_clipped = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantage[batch]

                actor_loss = - torch.min(weighted_probs, weighted_clipped).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.NMD_opt.zero_grad()
                total_loss.backward()
                self.NMD_opt.step()

                loss = total_loss.item()

        return loss
    def save_weights(self, filename):
        torch.save(self.ppo_model.state_dict(), filename)
    
    def load_weights(self, filename):
        self.ppo_model(torch.load(filename))
    


Agent = ppo(env.observation_space.shape[0], env.action_space.n)

def run(agent, env, train=True):
    
    all_rewards = []
    reward_windows = deque(maxlen=100)
    losses = []
    #loss_window = deque(maxlen=100)
    if train:
        a = 0
        for i in tqdm(range(1, 501)):
            
            Agent.memory.clear_buffer()
            score = 0
            state = env.reset()
            done = False
            entropy = 0
            last_reward = 1
            
            while not done:

                a += 1
                print(a)
                dist, value = Agent.get_model_out(state, last_reward)
                
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                
                next_state, reward, done, _ = env.step(action.item())

                Agent.memory.push(state, log_prob, value, action, reward, done)
                score += reward

                state = next_state
                last_reward = reward
            all_rewards.append(score)
            reward_windows.append(score)

            loss = Agent.update()
            losses.append(loss)
            print("last episode reward:{}|mean reward:{}".format(score, np.mean(reward_windows)))

            if i > 1:
                if np.mean(reward_windows) > 1500:
                    print("training complete!")
                    break
    """else:
        state = env.reset()
        counter = 0
        eps = 0.001
        while counter != 10:
            action, value, log_prob, dist = Agent.action(state)
            time.sleep(0.01)
            env.render()
            next_state, r, d, i = env.step(action)
            state = next_state
            if d:
                counter += 1
                state = env.reset()"""
    return all_rewards, losses, i


#Agent.load_weights('mario_ppo_1k.pt')
rewards, losses, index = run(Agent, env)

filename = 'Mario_ppo_5_no_lstm.pt'

Agent.save_weights(filename)

rewards_filename = 'Rewards_ppo_5_no_lstm.npy'
losses_filename = 'Losses_ppo_5_no_lstm.npy'

with open(rewards_filename, 'wb') as f:
  np.save(f, rewards)

with open(losses_filename, 'wb') as f:
  np.save(f, losses)