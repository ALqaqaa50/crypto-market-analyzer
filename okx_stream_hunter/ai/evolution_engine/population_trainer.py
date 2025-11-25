import numpy as np
import logging
from typing import Dict, List, Callable
import copy
import random

logger = logging.getLogger(__name__)


class PopulationBasedTrainer:
    def __init__(self, population_size: int = 20, exploitation_interval: int = 100,
                 truncation_percent: float = 0.2, resample_probability: float = 0.25):
        
        self.population_size = population_size
        self.exploitation_interval = exploitation_interval
        self.truncation_percent = truncation_percent
        self.resample_probability = resample_probability
        
        self.population = []
        self.step = 0
        
        logger.info(f"PopulationBasedTrainer initialized: pop_size={population_size}, "
                   f"exploit_interval={exploitation_interval}")
    
    def initialize_population(self, model_generator: Callable, hyperparam_space: Dict):
        self.population = []
        
        for i in range(self.population_size):
            hyperparams = {}
            for key, (low, high) in hyperparam_space.items():
                if isinstance(low, int):
                    hyperparams[key] = random.randint(low, high)
                else:
                    hyperparams[key] = random.uniform(low, high)
            
            member = {
                'id': i,
                'model': model_generator(hyperparams),
                'hyperparams': hyperparams,
                'performance': 0.0,
                'steps': 0
            }
            
            self.population.append(member)
        
        logger.info(f"Population initialized with {self.population_size} members")
    
    def train_step(self, train_func: Callable, eval_func: Callable):
        for member in self.population:
            train_func(member['model'], member['hyperparams'])
            member['performance'] = eval_func(member['model'])
            member['steps'] += 1
        
        self.step += 1
        
        if self.step % self.exploitation_interval == 0:
            self._exploit_and_explore()
    
    def _exploit_and_explore(self):
        self.population.sort(key=lambda x: x['performance'], reverse=True)
        
        truncation_size = int(self.population_size * self.truncation_percent)
        bottom_members = self.population[-truncation_size:]
        top_members = self.population[:truncation_size]
        
        for bottom in bottom_members:
            top = random.choice(top_members)
            
            bottom['model'] = copy.deepcopy(top['model'])
            bottom['hyperparams'] = copy.deepcopy(top['hyperparams'])
            
            for key, value in bottom['hyperparams'].items():
                if random.random() < self.resample_probability:
                    if isinstance(value, int):
                        bottom['hyperparams'][key] = int(value * random.uniform(0.8, 1.2))
                    else:
                        bottom['hyperparams'][key] = value * random.uniform(0.8, 1.2)
        
        logger.info(f"Step {self.step}: Exploit & Explore completed, "
                   f"best_perf={self.population[0]['performance']:.4f}")
    
    def get_best_model(self):
        best = max(self.population, key=lambda x: x['performance'])
        return best['model'], best['hyperparams'], best['performance']
    
    def get_population_stats(self) -> Dict:
        perfs = [m['performance'] for m in self.population]
        return {
            'best': max(perfs),
            'worst': min(perfs),
            'mean': np.mean(perfs),
            'std': np.std(perfs),
            'step': self.step
        }
