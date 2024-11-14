import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.learning_rate = 1.0  # Initialize learning rate
        
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        
        # Quantum rotation gate operation with dynamic learning rate
        rotation_angle = np.arctan(np.sqrt(np.log(1 + np.arange(self.dim)) / np.arange(1, self.dim + 1)))
        for i in range(1, self.budget):
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    self.population[i][j] = np.cos(rotation_angle[j] * self.learning_rate) * self.population[i][j] - np.sin(rotation_angle[j] * self.learning_rate) * elite[j]
                else:
                    self.population[i][j] = np.sin(rotation_angle[j] * self.learning_rate) * self.population[i][j] + np.cos(rotation_angle[j] * self.learning_rate) * elite[j]
        
        self.learning_rate *= 0.95  # Update learning rate
        
    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]
        
        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]
        
        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution