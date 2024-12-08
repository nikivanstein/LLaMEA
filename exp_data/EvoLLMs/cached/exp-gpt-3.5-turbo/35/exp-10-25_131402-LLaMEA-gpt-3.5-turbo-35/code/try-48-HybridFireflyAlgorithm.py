import numpy as np

class HybridFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.5
        self.beta = 0.2
        self.gamma = 0.1

    def __call__(self, func):
        population = np.random.uniform(low=-5.0, high=5.0, size=(10 * self.dim, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - 10 * self.dim):
            for i in range(len(population)):
                for j in range(len(population)):
                    if fitness[j] < fitness[i]:
                        distance = np.linalg.norm(population[j] - population[i])
                        attractiveness = 1 / (1 + self.alpha * distance**2)
                        population[i] += self.beta * attractiveness * (population[j] - population[i]) + self.gamma * np.random.normal(0, 1, self.dim)
                        fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness