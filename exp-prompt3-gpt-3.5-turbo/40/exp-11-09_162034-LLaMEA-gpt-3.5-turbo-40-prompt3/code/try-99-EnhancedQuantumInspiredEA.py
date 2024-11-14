import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        # Dynamic adjustment of rotation angles based on fitness values
        for i in range(1, self.budget):
            for j in range(self.dim):
                rotation_angle = np.arctan(np.sqrt(np.log(1 + np.abs(fitness_values[i] - fitness_values[sorted_indices[0]])) / np.abs(fitness_values[i] - fitness_values[sorted_indices[0]])))
                if np.random.rand() < 0.5:
                    self.population[i][j] = np.cos(rotation_angle) * self.population[i][j] - np.sin(rotation_angle) * elite[j]
                else:
                    self.population[i][j] = np.sin(rotation_angle) * self.population[i][j] + np.cos(rotation_angle) * elite[j]
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution