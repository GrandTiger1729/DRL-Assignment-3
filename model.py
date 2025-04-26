import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *

import logging

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate the output size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output_size = self.conv(dummy_input).shape[1]
            
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        
    def forward(self, x):
        x = self.conv(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_value = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_value
    
class ReplayBuffer:
    def __init__(self, capacity, state_dim, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.full = False

        self.states = torch.zeros((capacity, *state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.position] = torch.tensor([action], dtype=torch.int64, device=self.device)
        self.rewards[self.position] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[self.position] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.position] = torch.tensor([done], dtype=torch.float32, device=self.device)

        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.position
        indices = np.random.choice(max_index, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.capacity if self.full else self.position


class DQNAgent:
    """DQN Agent for reinforcement learning."""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.epsilon_decay = (EPSILON_END - EPSILON_START) / EPSILON_DECAY

        self.dqn = DQN(state_dim, action_dim).to(DEVICE)
        self.dqn_target = DQN(state_dim, action_dim).to(DEVICE)
        
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY, state_dim, device=DEVICE)
        
        self.update_count = 0
        
    def select_action(self, state, deterministic=False):
        if deterministic or np.random.rand() > self.epsilon:
            with torch.no_grad():
                q_values = self.dqn(torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.action_dim)
        return action
    
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        q_values = self.dqn(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.dqn(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.dqn_target(next_states).gather(1, next_actions)
            expected_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(EPSILON_END, self.epsilon + self.epsilon_decay)
        
        # Perform a hard update to the target network
        self.update_count += 1
        if self.update_count % UPDATE_TARGET_EVERY == 0:
            logging.info(f"Updating target network at step {self.update_count}")
            self.dqn_target.load_state_dict(self.dqn.state_dict())

