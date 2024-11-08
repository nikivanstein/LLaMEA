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
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            distances = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * distances)
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            better_indexes = np.argwhere(fitness < fitness.reshape(-1, 1))
            better_diff = population[better_indexes[:, 1]] - population[better_indexes[:, 0]]
            better_attractiveness = attractiveness_matrix[better_indexes[:, 0], better_indexes[:, 1]]
            
            population += np.sum(better_attractiveness[:, :, np.newaxis] * better_diff[:, :, np.newaxis], axis=0) + steps
            population = np.clip(population, -5.0, 5.0)
            fitness = np.array([func(individual) for individual in population])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]