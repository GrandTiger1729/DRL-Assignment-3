import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *
import logging

class DQN(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate the output size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim)
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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.full = False
        self.n_steps = N_STEPS  # Number of steps for n-step return
        
        self.states = torch.zeros((capacity, *state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        
        # Initialize priorities to zero
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        # Initialize beta for IS weights annealing
        self.beta = BETA_START
        
        # Temporary buffer for n-step returns
        self.n_step_buffer = []

    def _get_n_step_info(self):
        """Compute n-step returns and get relevant information."""
        state, action = self.n_step_buffer[0][:2]
        
        # Calculate n-step return (discounted sum of rewards)
        n_step_reward = 0
        for i in range(self.n_steps):
            if i >= len(self.n_step_buffer):
                break
                
            n_step_reward += (GAMMA ** i) * self.n_step_buffer[i][2]
            
            # If a terminal state is found, break the loop
            if self.n_step_buffer[i][4]:  # done flag
                next_state = self.n_step_buffer[i][3]  # next_state of the terminal transition
                done = True
                break
        else:
            # If we didn't break out early, use the state after n steps
            next_state = self.n_step_buffer[min(self.n_steps, len(self.n_step_buffer)) - 1][3]
            done = False
            
        return state, action, n_step_reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        # Add the transition to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Process n-step return only when buffer has enough transitions
        if len(self.n_step_buffer) < self.n_steps and not done:
            return
            
        # Get n-step transitions
        n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
        
        # Store the n-step transition in the main buffer
        self.states[self.position] = torch.tensor(n_step_state, dtype=torch.float32, device=self.device)
        self.actions[self.position] = torch.tensor([n_step_action], dtype=torch.int64, device=self.device)
        self.rewards[self.position] = torch.tensor([n_step_reward], dtype=torch.float32, device=self.device)
        self.next_states[self.position] = torch.tensor(n_step_next_state, dtype=torch.float32, device=self.device)
        self.dones[self.position] = torch.tensor([n_step_done], dtype=torch.float32, device=self.device)
        
        # Set max priority for new transition (if empty, default to 1.0)
        max_priority = self.priorities.max() if (self.position > 0 or self.full) else 1.0
        self.priorities[self.position] = max_priority if max_priority > 0 else 1.0

        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True
            
        # Remove the oldest transition (keep buffer size equal to n_steps)
        if done:
            # Clear buffer if episode ends
            self.n_step_buffer = []
        else:
            # Remove oldest transition
            self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.position
        # Compute probabilities based on priorities raised to ALPHA
        prios = self.priorities[:max_index] ** ALPHA
        prob = prios / prios.sum()
        
        # Sample indices with probability proportional to priority
        indices = np.random.choice(max_index, batch_size, p=prob)
        
        # Compute importance sampling (IS) weights using the current beta value
        total = max_index
        weights = (total * prob[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize for stability
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        # Update priorities with new TD errors (add PRIORITY_EPS for numerical stability)
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + PRIORITY_EPS

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
        # Use PrioritizedReplayBuffer instead of uniform ReplayBuffer
        self.replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_CAPACITY, state_dim, device=DEVICE)
        
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
        
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(BATCH_SIZE)
        
        q_values = self.dqn(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.dqn(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.dqn_target(next_states).gather(1, next_actions)
            expected_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        # Calculate TD errors for updating priorities
        td_errors = torch.abs(q_values - expected_q_values).detach().cpu().numpy().squeeze()
        
        # Multiply the loss by the IS weights
        loss = (weights * F.smooth_l1_loss(q_values, expected_q_values, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Anneal beta towards 1.0 over BETA_FRAMES steps
        self.replay_buffer.beta = min(1.0, self.replay_buffer.beta + (1.0 - BETA_START) / BETA_FRAMES)
        
        # Update epsilon
        self.epsilon = max(EPSILON_END, self.epsilon + self.epsilon_decay)
        
        # Perform a hard update to the target network
        self.update_count += 1
        if self.update_count % UPDATE_TARGET_EVERY == 0:
            logging.info(f"Updating target network at step {self.update_count}")
            self.dqn_target.load_state_dict(self.dqn.state_dict())

