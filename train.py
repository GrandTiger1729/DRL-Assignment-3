import numpy as np
import torch
from tqdm import tqdm

from mario_env import make_env
from model import DQN, DQNAgent
from constants import *

import logging

def set_seed(seed, env):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def select_action(dqn, obs):
    with torch.no_grad():
        q_values = dqn(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        action = torch.argmax(q_values).item()
    return action
    
def evaluate(env, dqn, num_episodes=VALIDATION_EPISODES):
    scores = []
    for _ in range(num_episodes):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = select_action(dqn, obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            score += reward
        scores.append(score)
    return np.mean(scores)

def train(env, agent: DQNAgent, num_episodes=NUM_EPISODES):
    best_validation_score = -np.inf
    
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        score = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            steps += 1
            
            # Store transition in replay buffer
            agent.store(obs, action, reward, next_obs, done)
            # Train the agent
            if steps % TRAIN_EVERY == 0:
                agent.train()
            
            obs = next_obs
            score += reward
        
        # Validation
        if (episode + 1) % VALIDATE_EVERY == 0:
            avg_score = evaluate(env, agent.dqn)
            # print(f"Episode {episode}, Average Validation Score: {avg_score}")
            logging.info(f"Episode {episode + 1}, Average Validation Score: {avg_score}")
            if avg_score > best_validation_score:
                best_validation_score = avg_score
                torch.save(agent.dqn.state_dict(), "mario-dqn.pth")
                # print(f"New best model saved with score: {best_validation_score}")
                logging.info(f"New best model saved with score: {best_validation_score}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set up the environment
    env = make_env()
    set_seed(SEED, env)
    
    # Train the agent
    agent = DQNAgent(state_dim=(FRAME_LENGTH, FEATURE_SIZE, FEATURE_SIZE), action_dim=env.action_space.n)
    train(env, agent, num_episodes=NUM_EPISODES)
    env.close()
    
    # # Evaluate the agent
    # DEVICE = torch.device("cpu")
    # env = make_env()
    # dqn = DQN((FRAME_LENGTH, FEATURE_SIZE, FEATURE_SIZE), env.action_space.n).to(DEVICE)
    # dqn.load_state_dict(torch.load("mario-dqn.pth", map_location=DEVICE))
    # dqn.eval()
    # mean_score = evaluate(env, dqn, num_episodes=10)
    # print(f"Mean score over 10 episodes: {mean_score}")
    # env.close()