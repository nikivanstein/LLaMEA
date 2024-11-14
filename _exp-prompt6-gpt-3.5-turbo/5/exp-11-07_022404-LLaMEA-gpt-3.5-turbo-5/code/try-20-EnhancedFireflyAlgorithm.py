import numpy as np

class EnhancedFireflyAlgorithm:
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
            dist_matrix = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * dist_matrix)
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            better_indexes = np.argwhere(fitness < fitness[:, np.newaxis])
            diff_pop = population[better_indexes[:, 1]] - population[better_indexes[:, 0]]
            weighted_diff = np.sum(attractiveness_matrix[better_indexes[:, 0], better_indexes[:, 1]][:, np.newaxis] * diff_pop, axis=1)
            
            population += np.where(fitness < fitness[:, np.newaxis], weighted_diff[:, np.newaxis], 0) + steps
            population = np.clip(population, -5.0, 5.0)
            fitness = np.array([func(individual) for individual in population])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]