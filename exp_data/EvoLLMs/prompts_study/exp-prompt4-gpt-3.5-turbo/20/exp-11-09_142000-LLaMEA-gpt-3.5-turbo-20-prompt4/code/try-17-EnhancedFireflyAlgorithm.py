import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim, elite_ratio=0.1):
        self.budget = budget
        self.dim = dim
        self.elite_ratio = elite_ratio
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def attractiveness(self, light_intensity, distance):
        beta = 1
        return light_intensity / (1 + beta * distance)

    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                self.population[idx] += alpha * np.exp(-beta_min * distance) * (self.population[i] - self.population[idx])

    def __call__(self, func):
        elitism_count = int(self.budget * self.elite_ratio)
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i)
            sorted_indices = np.argsort([func(ind) for ind in self.population])
            elites = self.population[sorted_indices[:elitism_count]]
            self.population[elitism_count:] = self.population[np.random.choice(elitism_count, self.budget - elitism_count, replace=True)]
            self.population[:elitism_count] = elites
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]