import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import TimeLimit

from collections import deque
import numpy as np
import cv2

from constants import *

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(SkipFrame, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class EdgeDetectionObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(FEATURE_SIZE, FEATURE_SIZE), dtype=np.float32
        )

    def observation(self, obs: np.ndarray):
        obs = obs[31:217, 0:248]
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(obs, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
        obs = cv2.Canny(obs, 100, 200)
        obs = obs.astype(np.float32) / 255.0
        return obs

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, stack_size=4):
        super(FrameStack, self).__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(stack_size, *env.observation_space.shape), dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return np.stack(list(self.frames), axis=0)

    def observation(self, obs):
        self.frames.append(obs)
        return np.stack(list(self.frames), axis=0)

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=SKIP)
    env = EdgeDetectionObservationWrapper(env)
    env = FrameStack(env, stack_size=FRAME_LENGTH)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODES_STEPS)
    return env