import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def attractiveness(self, light_intensity, distance):
        beta = 1
        return light_intensity / (1 + beta * distance)

    def levy_flight(self, step_size=0.1):
        return np.random.standard_cauchy(size=self.dim) * step_size / (np.abs(np.random.normal()) ** (1 / self.dim))

    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                step = alpha * np.exp(-beta_min * distance) * (self.population[i] - self.population[idx])
                step += self.levy_flight()
                self.population[idx] += step

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]