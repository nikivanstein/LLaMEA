import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def levy_flight(self, current_position, best_position):
        beta = 1.5
        step_size = (np.random.gamma(shape=1.5, scale=1.0, size=self.dim) ** (1.0 / beta)) * (current_position - best_position)
        return current_position + np.random.normal(0, 1, self.dim) * step_size
    
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        for i in range(1, self.budget):
            self.population[i] = self.levy_flight(self.population[i], elite)
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution