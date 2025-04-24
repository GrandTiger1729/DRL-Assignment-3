import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import cv2
from gym.wrappers import FrameStack

from model import clip_feature, ReplayBuffer, DQN, DQNAgent

def validate(validation_episodes):
    avg_score = 0
    
    for _ in range(validation_episodes):
        obs_stack = deque(maxlen=4)
        obs = clip_feature(env.reset())
        for _ in range(4):
            obs_stack.append(deepcopy(obs))
        score = 0
        
        done = False 
        while not done:
            action = agent.get_action(np.stack(list(reversed(obs_stack))), deterministic=True)
            next_obs, reward, done, info = env.step(action)

            next_obs = clip_feature(next_obs)
            obs_stack.append(next_obs)
            obs = next_obs
            score += reward
        avg_score += score / validation_episodes
        
    return avg_score

if __name__ == "__main__":
    # Set random seed for reproducibility
    SEED = 4949
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Create environment
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import RIGHT_ONLY, COMPLEX_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    import pickle
    from tqdm import tqdm
    from collections import deque
    from copy import deepcopy

    # Initialize agent
    # agent = DQNAgent(state_size=(4, 84, 84), action_size=env.action_space.n)
    with open("mario-dqn-agent.pkl", "rb") as f:
        agent: DQNAgent = pickle.load(f)
    
    # Training loop
    num_episodes = 10000
    scores = []
    
    validation_episodes = 10
    validation_interval = 1
    validation_scores = []
    best_validation_score = -np.inf
    
    for episode in tqdm(range(num_episodes)):
        obs_stack = deque(maxlen=4)
        obs = clip_feature(env.reset())
        for _ in range(4):
            obs_stack.append(deepcopy(obs))
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(np.stack(list(reversed(obs_stack))))
            
            next_obs, reward, done, info = env.step(action)
            next_obs = clip_feature(next_obs)            
            # Add to replay buffer
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            loss = agent.train()

            obs_stack.append(next_obs)
            obs = next_obs
            total_reward += reward

        scores.append(total_reward)
        
        if (episode + 1) % validation_interval == 0:
            
            avg_score = validate(validation_episodes)
            print(f"Episode {episode + 1}, Reward: {avg_score}")
            validation_scores.append(avg_score)
            
            # Dump model if it is the best
            if avg_score > best_validation_score:
                best_validation_score = avg_score
                print(f"New best model found at episode {episode + 1} with score {avg_score}")
                with open("mario-dqn-agent.pkl", "wb") as f:
                    pickle.dump(agent, f)