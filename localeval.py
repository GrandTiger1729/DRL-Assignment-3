import numpy as np

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from constants import *
from student_agent import Agent

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def evaluate(dqn, num_episodes=10):
    scores = []
    for _ in range(num_episodes):
        obs = env.reset()
        score = 0
        done = False
        agent = Agent(dqn)
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            score += reward
        scores.append(score)
    return np.mean(scores)