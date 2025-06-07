import numpy as np
from typing import Callable, Tuple

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        fitness_func: Callable[[np.ndarray], float],
        mutation_rate: float = 0.1
    ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_func = fitness_func
        self.mutation_rate = mutation_rate
        
        # Initialize population with reasonable resource values
        self.population = np.zeros((population_size, chromosome_length))
        for i in range(population_size):
            self.population[i] = [
                np.random.randint(100, 1000),  # CPU (millicores)
                np.random.randint(128, 1024),  # Memory (Mi)
                np.random.randint(1, 5)        # Replicas
            ]
            
    def _select_parents(self, fitness_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select parents using tournament selection"""
        tournament_size = 3
        parent1_idx = parent2_idx = 0
        
        for _ in range(tournament_size):
            idx = np.random.randint(0, self.population_size)
            if fitness_scores[idx] > fitness_scores[parent1_idx]:
                parent1_idx = idx
                
        for _ in range(tournament_size):
            idx = np.random.randint(0, self.population_size)
            if idx != parent1_idx and fitness_scores[idx] > fitness_scores[parent2_idx]:
                parent2_idx = idx
                
        return self.population[parent1_idx], self.population[parent2_idx]
        
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover"""
        child = np.zeros(self.chromosome_length)
        for i in range(self.chromosome_length):
            if np.random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child
        
    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Apply mutation with self-adapting step sizes"""
        for i in range(self.chromosome_length):
            if np.random.random() < self.mutation_rate:
                if i == 0:  # CPU
                    step = np.random.randint(-100, 100)
                    chromosome[i] = max(100, min(1000, chromosome[i] + step))
                elif i == 1:  # Memory
                    step = np.random.randint(-128, 128)
                    chromosome[i] = max(128, min(1024, chromosome[i] + step))
                else:  # Replicas
                    step = np.random.randint(-1, 2)
                    chromosome[i] = max(1, min(5, chromosome[i] + step))
        return chromosome
        
    def evolve(self, generations: int = 50) -> Tuple[np.ndarray, float]:
        """Run the genetic algorithm for specified generations"""
        best_fitness = float('-inf')
        best_chromosome = None
        
        for generation in range(generations):
            # Evaluate fitness for all chromosomes
            fitness_scores = np.array([
                self.fitness_func(chromosome) 
                for chromosome in self.population
            ])
            
            # Keep track of best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_chromosome = self.population[max_fitness_idx].copy()
            
            # Create new population
            new_population = []
            
            # Elitism: keep best solution
            new_population.append(self.population[max_fitness_idx])
            
            # Generate rest of new population
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self._select_parents(fitness_scores)
                
                # Create child through crossover
                child = self._crossover(parent1, parent2)
                
                # Apply mutation
                child = self._mutate(child)
                
                new_population.append(child)
            
            # Update population
            self.population = np.array(new_population)
            
        return best_chromosome, best_fitness 