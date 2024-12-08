import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def adapt_rotation_angle(self, iteration):
        return np.arctan(np.sqrt(np.log(1 + np.arange(self.dim)) / (iteration + np.arange(1, self.dim + 1))))
    
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        for i in range(1, self.budget):
            rotation_angle = self.adapt_rotation_angle(i)
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    self.population[i][j] = np.cos(rotation_angle[j]) * self.population[i][j] - np.sin(rotation_angle[j]) * elite[j]
                else:
                    self.population[i][j] = np.sin(rotation_angle[j]) * self.population[i][j] + np.cos(rotation_angle[j]) * elite[j]
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for it in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution