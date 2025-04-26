import gym
import numpy as np
import cv2
import torch
from collections import deque
import random

from model import DQN

def clip_feature(obs):
    obs = obs[31:217, 0:248]
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = cv2.Canny(obs, 100, 200)
    obs = obs.astype(np.float32) / 255.0
    return obs

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
        self.obs_stack = deque(maxlen=4)
        self.steps = 0
        self.last_action = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN((4, 84, 84), self.action_space.n)
        self.dqn.load_state_dict(torch.load("mario-dqn.pth", weights_only=False, map_location=self.device))
        self.dqn.eval()
        self.dqn.to(self.device)
        
    def get_action(self, obs):
        obs = clip_feature(obs.copy())
        # padding the obs_stack to 4 frames
        while len(self.obs_stack) < 4:
            self.obs_stack.append(obs.copy())
        
        # only use 1/4 of the frames
        if self.steps % 4 != 0:
            return self.last_action
        
        self.obs_stack.append(obs.copy())
        state = np.stack(list(self.obs_stack), axis=0)
        with torch.no_grad():
            q_values = self.dqn(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            action = torch.argmax(q_values).detach().cpu().numpy().item()
            
        return action
        
    def act(self, obs):
        action = self.get_action(obs)
        self.last_action = action
        self.steps += 1
        return action