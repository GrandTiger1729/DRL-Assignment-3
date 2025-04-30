import torch

SKIP = 4
FRAME_LENGTH = 4
DEATH_PENALTY = 5
LIVE_AWARD = 30
BANNED_ACTIONS = [7, 8, 9, 10, 11]
ACTION_PENALTY = 50
FEATURE_SIZE = 84
MAX_EPISODES_STEPS = 3000

SEED = 89487
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
GAMMA = 0.63
BATCH_SIZE = 32

REPLAY_BUFFER_CAPACITY = 10000
ALPHA = 0.6              # exponent for prioritization
BETA_START = 0.4         # starting value of beta for importance sampling
PRIORITY_EPS = 1e-6
BETA_FRAMES = 100000  # Number of training steps over which beta is annealed from BETA_START to 1.0
N_STEPS = 3

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 100000
TRAIN_EVERY = 4
UPDATE_TARGET_EVERY = 10000

NUM_EPISODES = 10000
VALIDATE_EVERY = 10
VALIDATION_EPISODES = 1
CHECKPOINT_EVERY = 100