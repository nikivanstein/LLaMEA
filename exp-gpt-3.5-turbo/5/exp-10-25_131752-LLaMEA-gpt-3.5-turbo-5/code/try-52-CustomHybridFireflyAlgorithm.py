import numpy as np

class CustomHybridFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def levy_flight(dim):
            sigma1 = (np.math.gamma(1 + self.beta0) * np.sin(np.pi * self.beta0 / 2)) / (np.math.gamma((1 + self.beta0) / 2) * self.beta0 * 2 ** ((self.beta0 - 1) / 2))
            sigma2 = 1
            u = np.random.normal(0, sigma1, dim)
            v = np.random.normal(0, sigma2, dim)
            step = u / np.power(np.abs(v), 1 / self.beta0)
            return self.alpha * step

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        population = initialize_population()
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(population[i]) < func(population[j]):
                        step = levy_flight(self.dim)
                        population[i] += self.gamma * step
                        population[i] = np.clip(population[i], -5.0, 5.0)
        best_index = np.argmin([func(individual) for individual in population])
        return population[best_index]