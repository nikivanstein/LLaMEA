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
        
        for _ in range(self.budget):
            fitness = np.array([func(individual) for individual in population])
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(population[:, np.newaxis] - population, axis=2))
            steps = self.alpha * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            better_fitness = fitness < fitness[:, np.newaxis]
            better_indexes = np.where(better_fitness)
            attractiveness_updates = np.sum(attractiveness_matrix * better_fitness[:,:,np.newaxis] * (population[better_indexes[1]] - population[better_indexes[0]][:, np.newaxis]), axis=1)
            
            population += attractiveness_updates + steps
            population = np.clip(population, -5.0, 5.0)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]