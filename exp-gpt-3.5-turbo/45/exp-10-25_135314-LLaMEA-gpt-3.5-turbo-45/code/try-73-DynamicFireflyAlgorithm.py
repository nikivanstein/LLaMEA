import numpy as np

class DynamicFireflyAlgorithm:
    def __init__(self, budget, dim, alpha_min=0.1, alpha_max=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def attractiveness(self, x, y):
        return np.exp(-np.linalg.norm(x - y))

    def levy_flight(self, alpha):
        return np.random.standard_cauchy(size=self.dim) / (np.abs(np.random.uniform()) ** (1 / alpha))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for iteration in range(self.budget):
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (iteration / self.budget)
            for i in range(self.budget):
                for j in range(self.budget):
                    if fitness[j] < fitness[i]:
                        step = self.attractiveness(population[i], population[j]) * (population[j] - population[i])
                        population[i] += alpha * step + self.levy_flight(alpha)
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])
        
        return population[np.argmin(fitness)]