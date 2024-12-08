import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.alpha = 0.5
        self.beta_min = 0.2
        self.alpha_mult = 0.9  # Adaptive step size control

    def attractiveness(self, light_intensity, distance):
        beta = 1
        return light_intensity / (1 + beta * distance)

    def move_firefly(self, idx):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                step_size = self.alpha * np.exp(-self.beta_min * distance)
                self.population[idx] += step_size * (self.population[i] - self.population[idx])
                self.alpha *= self.alpha_mult  # Update step size

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]