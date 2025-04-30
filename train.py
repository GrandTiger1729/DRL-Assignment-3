import os
import numpy as np
import torch
from tqdm import tqdm

from mario_env import make_env
from model import DQN, DQNAgent
from constants import *
from localeval import evaluate

import logging

def set_seed(seed, env):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def validate(dqn, num_episodes=VALIDATION_EPISODES):
    return evaluate(dqn, num_episodes=num_episodes)

def save_checkpoint(agent, episode, best_validation_score, filename="checkpoint.pth"):
    checkpoint = {
        "episode": episode,
        "best_validation_score": best_validation_score,
        "dqn_state_dict": agent.dqn.state_dict(),
        "dqn_target_state_dict": agent.dqn_target.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
    }
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved at episode {episode}")

def load_checkpoint(agent, filename="checkpoint.pth"):
    checkpoint = torch.load(filename, map_location=DEVICE)
    agent.dqn.load_state_dict(checkpoint["dqn_state_dict"])
    agent.dqn_target.load_state_dict(checkpoint["dqn_target_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.epsilon = checkpoint["epsilon"]
    logging.info(f"Checkpoint loaded from episode {checkpoint['episode']}")
    return checkpoint["episode"], checkpoint["best_validation_score"]

def train(env, agent: DQNAgent, num_episodes, start_episode=0, best_validation_score=-np.inf):
    """
    Train the DQN agent on the environment.
    """
    for episode in tqdm(range(start_episode, num_episodes)):
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
        
        # Validation step
        if (episode + 1) % VALIDATE_EVERY == 0:
            avg_score = validate(agent.dqn)
            logging.info(f"Episode {episode + 1}, Average Validation Score: {avg_score}")
            if avg_score > best_validation_score:
                best_validation_score = avg_score
                torch.save(agent.dqn.state_dict(), "mario-dqn.pth")
                logging.info(f"New best model saved with score: {best_validation_score}")
        
        # Save checkpoint every CHECKPOINT_EVERY episodes
        if (episode + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(agent, episode + 1, best_validation_score)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set up the environment
    env = make_env()
    set_seed(SEED, env)
    
    # Create agent
    agent = DQNAgent(state_dim=(FRAME_LENGTH, FEATURE_SIZE, FEATURE_SIZE), action_dim=env.action_space.n)
    
    # Check for an existing checkpoint to resume training
    start_episode = 0
    best_validation_score = -np.inf
    if os.path.exists("checkpoint.pth"):
        start_episode, best_validation_score = load_checkpoint(agent)
        logging.info(f"Resuming training from episode {start_episode}, best validation score: {best_validation_score}")
    
    train(env, agent, num_episodes=NUM_EPISODES, start_episode=start_episode, best_validation_score=best_validation_score)
    env.close()