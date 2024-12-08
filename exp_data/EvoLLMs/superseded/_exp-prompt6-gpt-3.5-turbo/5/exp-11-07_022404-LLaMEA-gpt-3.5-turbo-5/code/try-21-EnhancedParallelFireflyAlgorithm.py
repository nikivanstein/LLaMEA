import numpy as np

class EnhancedParallelFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.1
        self.beta_min = 0.2
        self.gamma = 1.0

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)  # Vectorized fitness evaluation
        
        for _ in range(self.budget):
            norm_matrix = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * norm_matrix)
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            better_indexes = np.argwhere(fitness < fitness[:, np.newaxis]).reshape(-1, 2)
            deltas = population[better_indexes[:, 1]] - population[better_indexes[:, 0]]
            
            population += np.sum(attractiveness_matrix[:, better_indexes[:, 0]] * deltas[:, :, np.newaxis], axis=1) + steps
            population = np.clip(population, -5.0, 5.0)
            fitness = np.apply_along_axis(func, 1, population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]