from model import ReplayBuffer, DQN, DQNAgent
import torch

if __name__ == "__main__":
    import pickle
    
    with open("mario-dqn-agent.pkl", "rb") as f:
        agent: DQNAgent = pickle.load(f)
        
    torch.save(agent.dqn.state_dict(), "mario-dqn.pth")