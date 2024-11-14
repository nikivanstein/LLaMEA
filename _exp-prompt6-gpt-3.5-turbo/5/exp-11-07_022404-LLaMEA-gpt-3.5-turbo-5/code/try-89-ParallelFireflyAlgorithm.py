import numpy as np

class ParallelFireflyAlgorithm:
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
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(population[:, np.newaxis] - population, axis=2))
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            for i in range(self.population_size):
                better_indexes = np.where(fitness < fitness[i])
                population[i] += np.sum(attractiveness_matrix[i, better_indexes[0]][:, np.newaxis] * (population[better_indexes] - population[i]), axis=0) + steps[i]
                population[i] = np.clip(population[i], -5.0, 5.0)
                fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]