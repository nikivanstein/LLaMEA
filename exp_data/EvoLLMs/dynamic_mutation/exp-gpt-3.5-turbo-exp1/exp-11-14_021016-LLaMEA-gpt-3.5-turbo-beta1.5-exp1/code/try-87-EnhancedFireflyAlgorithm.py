import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def levy_flight(self):
        return np.random.standard_cauchy(size=self.dim) / np.random.gamma(1.5, 1.0, size=self.dim)

    def move_fireflies(alpha=1.0, beta=1.0, gamma=0.5, step_size=0.1):
        new_population = np.copy(self.population)
        for i in range(self.budget):
            for j in range(self.budget):
                if func(self.population[j]) < func(self.population[i]):
                    step = alpha * np.exp(-beta * np.linalg.norm(self.population[j] - self.population[i]) ** 2) * (self.population[j] - self.population[i]) + gamma * self.levy_flight()
                    new_population[i] += step_size * step
        self.population = new_population