import numpy as np

class ImprovedFireflyAlgorithm:
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
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        attractiveness = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(population[i] - population[j]))
                        step = self.alpha * (np.random.rand(self.dim) - 0.5)
                        population[i] += attractiveness * (population[j] - population[i]) + step
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]