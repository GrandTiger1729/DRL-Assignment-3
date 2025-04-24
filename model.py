import numpy as np
import random
import torch
import torch.nn as nn
import cv2

def clip_feature(obs: np.ndarray):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # convert to grayscale
    obs = obs[72:240, 24:192] # crop the image to 168x168
    obs = cv2.resize(obs, (84, 84))
    return obs

class ReplayBuffer:
    def __init__(self, capacity, feature_size=(84, 84), stack_length=4):
        if isinstance(feature_size, int):
            feature_size = [feature_size]
        assert isinstance(feature_size, (list, tuple)) and len(feature_size) > 0

        self.feature_buffer = np.zeros((capacity, *feature_size))
        self.action_buffer = np.zeros(capacity)
        self.reward_buffer = np.zeros(capacity)
        self.next_feature_buffer = np.zeros((capacity, *feature_size))
        self.done_buffer = np.zeros(capacity)

        self.capacity = capacity
        self.stack_length = stack_length

        self.size = 0
        self.buffer_index = 0

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done):
        index = self.buffer_index % self.capacity
        self.feature_buffer[index] = obs
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_feature_buffer[index] = next_obs
        self.done_buffer[index] = done

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        assert batch_size <= self.size
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        # stack the features to state and next_state from cyclic buffer
        states = np.zeros((batch_size, self.stack_length, *self.feature_buffer.shape[1:]), dtype=self.feature_buffer.dtype)
        next_states = np.zeros((batch_size, self.stack_length, *self.feature_buffer.shape[1:]), dtype=self.feature_buffer.dtype)
        for i, index in enumerate(indices):
            for j in range(self.stack_length):
                states[i, j] = self.feature_buffer[(index - j) % self.capacity]
                next_states[i, j] = self.next_feature_buffer[(index - j) % self.capacity]
        
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.done_buffer[indices]
        return states, actions, rewards, next_states, dones

class DQN(nn.Module):
    """Require input shape (batch_size, 4, 84, 84)"""
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.shape[0], -1) if len(x.shape) == 4 else x.view(-1)
        return self.fc(x)

class DQNAgent:
    def __init__(self, 
                 state_size = (4, 84, 84),
                 action_size = 12,
                 *,
                 gamma = 0.99,
                 lr = 1e-4,
                 replay_buffer_capacity = 100000,
                 batch_size = 32,
                 epsilon_start = 1.0,
                 epsilon_end = 0.02,
                 epsilon_decay = 200000,
                 target_update_freq = 1000,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = device

        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity, feature_size=state_size[1:])
        self.batch_size = batch_size

        self.dqn = DQN(action_size).to(self.device)
        self.target_dqn = DQN(action_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.target_update_freq = target_update_freq
        self.update_count = 0
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

    def get_action(self, state, deterministic=True):
        # Epsilon greedy for exploration or exploitation
        if random.random() < self.epsilon and not deterministic:
            action = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.dqn(torch.tensor(state, dtype=torch.float32, device=self.device))
                action = torch.argmax(q_values).detach().cpu().numpy().item()
        if not deterministic:
            self.epsilon = max(self.epsilon_end, self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay)
        return action

    def update(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size=self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.dqn(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_dqn(next_states).gather(1, next_actions).squeeze(1)
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update()

        return float(loss.detach())