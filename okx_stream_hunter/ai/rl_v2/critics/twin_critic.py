import numpy as np
import logging
from okx_stream_hunter.ai.rl_v2.critics.critic_network import CriticNetwork

logger = logging.getLogger(__name__)


class TwinCritic:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [512, 256, 128],
                 learning_rate: float = 0.001, use_gpu: bool = True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dims, learning_rate, use_gpu)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dims, learning_rate, use_gpu)
        
        logger.info("TwinCritic initialized with two independent Q-networks")
    
    def predict(self, state: np.ndarray, action: np.ndarray, use_min: bool = True):
        q1 = self.critic1.predict(state, action)
        q2 = self.critic2.predict(state, action)
        
        if use_min:
            return np.minimum(q1, q2)
        else:
            return q1, q2
    
    def train_step(self, states: np.ndarray, actions: np.ndarray, target_q: np.ndarray):
        loss1 = self.critic1.train_step(states, actions, target_q)
        loss2 = self.critic2.train_step(states, actions, target_q)
        
        return (loss1 + loss2) / 2
    
    def save(self, filepath1: str, filepath2: str):
        self.critic1.save(filepath1)
        self.critic2.save(filepath2)
    
    def load(self, filepath1: str, filepath2: str):
        self.critic1.load(filepath1)
        self.critic2.load(filepath2)
    
    def get_weights(self):
        return {
            'critic1': self.critic1.get_weights(),
            'critic2': self.critic2.get_weights()
        }
    
    def set_weights(self, weights: dict):
        self.critic1.set_weights(weights['critic1'])
        self.critic2.set_weights(weights['critic2'])
