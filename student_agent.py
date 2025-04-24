import gym
import numpy as np
import cv2
import torch
from collections import deque

from model import clip_feature, ReplayBuffer, DQN, DQNAgent

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
        self.obs_stack = deque(maxlen=4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dqn = DQN(self.action_space.n)
        self.dqn.load_state_dict(torch.load("mario-dqn.pth", weights_only=True))
        self.dqn.eval()
        self.dqn.to(self.device)
        
    def get_action(self, obs):
        # padding the obs_stack to 4 frames
        while len(self.obs_stack) < 4:
            self.obs_stack.append(clip_feature(obs.copy()))
        # add the new observation to the stack
        self.obs_stack.append(clip_feature(obs.copy()))
        
        state = np.stack(list(reversed(self.obs_stack)))
        with torch.no_grad():
            q_values = self.dqn(torch.tensor(state, dtype=torch.float32, device=self.device))
            action = torch.argmax(q_values).detach().cpu().numpy().item()
            
        return action
        
    def act(self, obs):
        action = self.get_action(obs)
        return action