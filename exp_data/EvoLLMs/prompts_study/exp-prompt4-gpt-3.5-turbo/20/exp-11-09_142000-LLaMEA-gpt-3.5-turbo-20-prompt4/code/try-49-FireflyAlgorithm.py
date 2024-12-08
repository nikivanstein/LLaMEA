import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def attractiveness(self, light_intensity, distance):
        beta = 1
        return light_intensity / (1 + beta * distance)

    def levy_flight(self, idx, alpha=0.5):
        gamma = 1.5
        sigma = (np.math.gamma(1 + gamma) * np.sin(np.pi * gamma / 2) / (np.math.gamma((1 + gamma) / 2) * gamma * 2 ** ((gamma - 1) / 2))) ** (1 / gamma)
        s = np.random.normal(0, sigma, self.dim)
        levy = 0.01 * s / (np.abs(s) ** (1 + gamma))
        self.population[idx] += alpha * levy

    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                self.population[idx] += alpha * np.exp(-beta_min * distance) * (self.population[i] - self.population[idx])

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i)
                        self.levy_flight(i)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]