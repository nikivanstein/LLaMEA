import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def quantum_rotation(self, population):
        return population * np.exp(1j * np.pi/2)
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        for _ in range(self.budget):
            rotated_population = self.quantum_rotation(population)
            fitness_values = np.apply_along_axis(func, 1, rotated_population)
            best_idx = np.argmin(fitness_values)
            population[0] = rotated_population[best_idx]
        
        return population[0]