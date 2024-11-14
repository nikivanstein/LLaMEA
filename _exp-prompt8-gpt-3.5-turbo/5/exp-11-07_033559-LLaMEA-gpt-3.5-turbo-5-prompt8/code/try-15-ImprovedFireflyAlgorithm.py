import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            for i in range(self.budget):
                if fitness[i] < fitness.min():
                    best_idx = i
                    best_fitness = fitness[i]
                    best_individual = population[i]

            for j in range(self.budget):
                if fitness[j] < best_fitness:
                    r = np.linalg.norm(population[best_idx] - population[j])
                    attractiveness = self.beta * np.exp(-self.gamma * r**2)
                    population[best_idx] += self.alpha * (attractiveness * (population[j] - population[best_idx])) + np.random.uniform(-1, 1, self.dim)
                    fitness[best_idx] = func(population[best_idx])
        
        return population[best_idx]