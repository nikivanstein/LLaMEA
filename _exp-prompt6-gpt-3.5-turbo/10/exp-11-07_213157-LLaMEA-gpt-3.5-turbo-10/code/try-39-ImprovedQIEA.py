import numpy as np
from joblib import Parallel, delayed

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def quantum_rotation(self, x, alpha):
        return x * np.exp(1j * alpha)
    
    def _evaluate_fitness(self, func, population):
        return [func(individual) for individual in population]
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = np.zeros(self.dim)
        
        for _ in range(self.budget):
            rotated_population = [self.quantum_rotation(individual, np.pi/2) for individual in population]
            fitness_values = Parallel(n_jobs=-1)(delayed(self._evaluate_fitness)(func, rotated_population) for _ in range(self.budget))
            best_idx = np.argmin(fitness_values)
            best_solution = rotated_population[best_idx]
            population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        return best_solution