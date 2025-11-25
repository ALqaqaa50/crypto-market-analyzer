import numpy as np
import logging
from typing import Tuple, List
import threading

logger = logging.getLogger(__name__)


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
    
    def add(self, priority: float, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get(self, s: float) -> Tuple[int, float, object]:
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 0.01):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.lock = threading.Lock()
        
        logger.info(f"PrioritizedReplayBuffer initialized: capacity={capacity}, alpha={alpha}, beta={beta}")
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        with self.lock:
            experience = (state, action, reward, next_state, done)
            priority = self.max_priority ** self.alpha
            self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray, np.ndarray, List[int]]:
        with self.lock:
            batch = []
            indices = []
            priorities = []
            segment = self.tree.total_priority / batch_size
            
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                
                idx, priority, data = self.tree.get(s)
                
                if data is None:
                    continue
                
                priorities.append(priority)
                batch.append(data)
                indices.append(idx)
            
            if len(batch) == 0:
                return None, None, None, None, None, None, []
            
            sampling_probabilities = np.array(priorities) / self.tree.total_priority
            is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
            is_weights /= is_weights.max()
            
            states = np.array([exp[0] for exp in batch])
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)
            next_states = np.array([exp[3] for exp in batch])
            dones = np.array([exp[4] for exp in batch]).reshape(-1, 1)
            
            return states, actions, rewards, next_states, dones, is_weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        with self.lock:
            for idx, td_error in zip(indices, td_errors):
                priority = (abs(td_error) + self.epsilon) ** self.alpha
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.size
    
    def is_ready(self, min_size: int = 1000) -> bool:
        return len(self) >= min_size
