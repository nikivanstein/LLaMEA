import numpy as np

class FireflyAlgorithmLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def attractiveness(self, light_intensity, distance):
        beta = 1
        return light_intensity / (1 + beta * distance)

    def levy_flight(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                step = alpha * np.random.standard_cauchy(size=self.dim) / np.power(np.linalg.norm(self.population[i] - self.population[idx]), beta_min)
                self.population[idx] += step

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.levy_flight(i)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]