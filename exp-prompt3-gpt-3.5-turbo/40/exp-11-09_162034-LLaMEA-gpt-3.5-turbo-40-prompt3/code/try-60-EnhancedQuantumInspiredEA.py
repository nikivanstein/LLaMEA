import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def evolve(self, fitness_values, iteration):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        # Dynamic learning rate based on iteration number
        learning_rate = 2.0 / (1.0 + np.exp(-4.0 * iteration / self.budget)) - 1.0
        
        for i in range(1, self.budget):
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    self.population[i][j] = np.cos(learning_rate * rotation_angle[j]) * self.population[i][j] - np.sin(learning_rate * rotation_angle[j]) * elite[j]
                else:
                    self.population[i][j] = np.sin(learning_rate * rotation_angle[j]) * self.population[i][j] + np.cos(learning_rate * rotation_angle[j]) * elite[j]
    
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for iteration in range(self.budget):
            self.evolve(fitness_values, iteration)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution