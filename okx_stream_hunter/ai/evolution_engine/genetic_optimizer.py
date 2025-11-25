import numpy as np
import logging
from typing import Dict, List, Callable
import copy
import random

logger = logging.getLogger(__name__)


class Individual:
    def __init__(self, genes: Dict, fitness: float = 0.0):
        self.genes = genes
        self.fitness = fitness
        self.age = 0
    
    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.2):
        mutated_genes = copy.deepcopy(self.genes)
        
        for key, value in mutated_genes.items():
            if random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    if isinstance(value, int):
                        mutated_genes[key] = int(value * (1 + np.random.normal(0, mutation_scale)))
                    else:
                        mutated_genes[key] = value * (1 + np.random.normal(0, mutation_scale))
                elif isinstance(value, list):
                    if random.random() < 0.5:
                        idx = random.randint(0, len(value) - 1)
                        if isinstance(value[idx], (int, float)):
                            if isinstance(value[idx], int):
                                value[idx] = int(value[idx] * (1 + np.random.normal(0, mutation_scale)))
                            else:
                                value[idx] = value[idx] * (1 + np.random.normal(0, mutation_scale))
        
        return Individual(mutated_genes)
    
    def crossover(self, other: 'Individual') -> tuple:
        child1_genes = {}
        child2_genes = {}
        
        for key in self.genes.keys():
            if random.random() < 0.5:
                child1_genes[key] = self.genes[key]
                child2_genes[key] = other.genes[key]
            else:
                child1_genes[key] = other.genes[key]
                child2_genes[key] = self.genes[key]
        
        return Individual(child1_genes), Individual(child2_genes)


class GeneticOptimizer:
    def __init__(self, population_size: int = 50, elite_size: int = 10,
                 mutation_rate: float = 0.1, mutation_scale: float = 0.2,
                 tournament_size: int = 5, max_generations: int = 100):
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Individual = None
        self.fitness_history = []
        
        logger.info(f"GeneticOptimizer initialized: pop_size={population_size}, elite={elite_size}, "
                   f"mutation_rate={mutation_rate}")
    
    def initialize_population(self, gene_template: Dict):
        self.population = []
        
        for _ in range(self.population_size):
            genes = {}
            for key, value in gene_template.items():
                if isinstance(value, int):
                    genes[key] = random.randint(max(1, value // 2), value * 2)
                elif isinstance(value, float):
                    genes[key] = value * random.uniform(0.5, 1.5)
                elif isinstance(value, list):
                    genes[key] = [v * random.uniform(0.5, 1.5) if isinstance(v, (int, float)) else v 
                                 for v in value]
                elif isinstance(value, str):
                    genes[key] = value
                else:
                    genes[key] = value
            
            self.population.append(Individual(genes))
        
        logger.info(f"Population initialized with {self.population_size} individuals")
    
    def evaluate_population(self, fitness_func: Callable[[Dict], float]):
        for individual in self.population:
            try:
                individual.fitness = fitness_func(individual.genes)
                individual.age += 1
            except Exception as e:
                logger.error(f"Error evaluating individual: {e}")
                individual.fitness = -np.inf
        
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.population[0])
        
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': self.population[0].fitness,
            'avg_fitness': avg_fitness
        })
        
        logger.info(f"Gen {self.generation}: Best={self.population[0].fitness:.4f}, "
                   f"Avg={avg_fitness:.4f}")
    
    def tournament_selection(self) -> Individual:
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def evolve(self, fitness_func: Callable[[Dict], float], generations: int = None) -> Dict:
        if generations is None:
            generations = self.max_generations
        
        for gen in range(generations):
            self.generation = gen
            
            self.evaluate_population(fitness_func)
            
            elite = self.population[:self.elite_size]
            
            offspring = []
            
            while len(offspring) < self.population_size - self.elite_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                if random.random() < 0.7:
                    child1, child2 = parent1.crossover(parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                child1 = child1.mutate(self.mutation_rate, self.mutation_scale)
                child2 = child2.mutate(self.mutation_rate, self.mutation_scale)
                
                offspring.append(child1)
                if len(offspring) < self.population_size - self.elite_size:
                    offspring.append(child2)
            
            self.population = elite + offspring
        
        return {
            'best_genes': self.best_individual.genes,
            'best_fitness': self.best_individual.fitness,
            'generations': self.generation + 1,
            'fitness_history': self.fitness_history
        }
    
    def get_best(self) -> tuple:
        return self.best_individual.genes, self.best_individual.fitness
    
    def save_checkpoint(self, filepath: str):
        import json
        checkpoint = {
            'generation': self.generation,
            'best_individual': {
                'genes': self.best_individual.genes,
                'fitness': self.best_individual.fitness
            },
            'population': [{'genes': ind.genes, 'fitness': ind.fitness, 'age': ind.age} 
                          for ind in self.population],
            'fitness_history': self.fitness_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        import json
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.generation = checkpoint['generation']
        self.best_individual = Individual(
            checkpoint['best_individual']['genes'],
            checkpoint['best_individual']['fitness']
        )
        
        self.population = [
            Individual(ind['genes'], ind['fitness'])
            for ind in checkpoint['population']
        ]
        
        for i, ind_data in enumerate(checkpoint['population']):
            self.population[i].age = ind_data['age']
        
        self.fitness_history = checkpoint['fitness_history']
        
        logger.info(f"Checkpoint loaded: {filepath}, generation={self.generation}")
