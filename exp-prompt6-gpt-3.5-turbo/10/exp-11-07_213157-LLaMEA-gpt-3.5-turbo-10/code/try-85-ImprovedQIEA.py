import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def quantum_rotation(self, x, alpha):
        return x * np.exp(1j * alpha)
    
    def roulette_selection(self, fitness_values):
        total_fitness = np.sum(fitness_values)
        probabilities = np.cumsum(fitness_values) / total_fitness
        selected_idx = np.searchsorted(probabilities, np.random.rand())
        return selected_idx
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = np.zeros(self.dim)
        
        for _ in range(self.budget):
            rotated_population = [self.quantum_rotation(individual, np.pi/2) for individual in population]
            fitness_values = [func(individual) for individual in rotated_population]
            
            best_idx = self.roulette_selection(fitness_values)
            best_solution = rotated_population[best_idx]
            
            population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        return best_solution