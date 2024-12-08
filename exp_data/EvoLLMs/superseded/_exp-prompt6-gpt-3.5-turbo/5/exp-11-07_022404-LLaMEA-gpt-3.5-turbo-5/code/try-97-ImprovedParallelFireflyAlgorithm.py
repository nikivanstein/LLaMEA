import numpy as np

class ImprovedParallelFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.1
        self.beta_min = 0.2
        self.gamma = 1.0

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array(list(map(func, population)))  # Vectorized fitness evaluation
        
        for _ in range(self.budget):
            euclidean_distances = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * euclidean_distances)
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            better_indexes = np.argwhere(fitness < fitness[:, np.newaxis])
            pop_diff = population[better_indexes[:, 1]] - population[better_indexes[:, 0]]
            attraction_sum = np.sum(attractiveness_matrix[better_indexes[:, 0], better_indexes[:, 1]][:, np.newaxis] * pop_diff, axis=0)
            
            for i in range(self.population_size):
                population[i] += attraction_sum[np.where(better_indexes[:, 1] == i)[0]].sum() + steps[i]
                population[i] = np.clip(population[i], -5.0, 5.0)
                fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]