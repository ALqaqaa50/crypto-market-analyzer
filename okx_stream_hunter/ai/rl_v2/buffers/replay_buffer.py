import numpy as np
import logging
from collections import deque
from typing import Tuple, List
import threading

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000, state_dim: int = 128, action_dim: int = 3):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.position = 0
        self.size = 0
        self.lock = threading.Lock()
        
        logger.info(f"ReplayBuffer initialized: capacity={capacity}, state_dim={state_dim}, action_dim={action_dim}")
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        with self.lock:
            self.states[self.position] = state
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.next_states[self.position] = next_state
            self.dones[self.position] = float(done)
            
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            if self.size < batch_size:
                batch_size = self.size
            
            indices = np.random.randint(0, self.size, size=batch_size)
            
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices]
            )
    
    def __len__(self):
        return self.size
    
    def is_ready(self, min_size: int = 1000) -> bool:
        return self.size >= min_size
    
    def clear(self):
        with self.lock:
            self.position = 0
            self.size = 0
            self.states.fill(0)
            self.actions.fill(0)
            self.rewards.fill(0)
            self.next_states.fill(0)
            self.dones.fill(0)
