import numpy as np

class PerformanceImprovedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha_min = 0.1
        self.alpha_max = 0.5
        self.beta_min = 0.2
        self.gamma = 1.0

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            attractiveness_matrix = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(population[:, np.newaxis] - population, axis=2))
            alphas = self.alpha_min + (self.alpha_max - self.alpha_min) * (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness))
            steps = alphas[:, np.newaxis] * (np.random.rand(self.population_size, self.dim) - 0.5)
            
            better_fitness = fitness < fitness[:, np.newaxis]
            population += np.sum(attractiveness_matrix[:, :, np.newaxis] * (population[better_fitness] - population[:, np.newaxis]), axis=1) + steps
            population = np.clip(population, -5.0, 5.0)
            fitness = np.array([func(individual) for individual in population])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]