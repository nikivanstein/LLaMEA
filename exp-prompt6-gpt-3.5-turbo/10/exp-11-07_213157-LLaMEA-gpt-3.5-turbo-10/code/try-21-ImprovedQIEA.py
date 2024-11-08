import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def quantum_rotation(self, x, alpha):
        return x * np.exp(1j * alpha)
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = population[0]  # Initialize best solution with the first individual
        
        for _ in range(1, self.budget):
            rotated_population = [self.quantum_rotation(individual, np.pi/2) for individual in population]
            fitness_values = [func(individual) for individual in rotated_population]
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < func(best_solution):
                best_solution = rotated_population[best_idx]
        
        return best_solution