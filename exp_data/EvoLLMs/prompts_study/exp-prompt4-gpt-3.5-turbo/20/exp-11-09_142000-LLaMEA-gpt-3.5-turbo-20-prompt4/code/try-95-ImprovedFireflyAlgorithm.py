import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def attractiveness(self, light_intensity, distance):
        beta = 1
        return light_intensity / (1 + beta * distance)

    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                self.population[idx] += alpha * np.exp(-beta_min * distance) * (self.population[i] - self.population[idx])

    def dynamic_adjustment(self, iteration, max_iter):
        alpha = 0.5 - iteration * (0.5 / max_iter)  # Dynamic adjustment of alpha
        beta_min = 0.2 + iteration * (0.8 / max_iter)  # Dynamic adjustment of beta_min
        return alpha, beta_min

    def __call__(self, func):
        max_iter = self.budget // 5  # Define maximum iterations based on budget
        for iter_count in range(max_iter):
            alpha, beta_min = self.dynamic_adjustment(iter_count, max_iter)
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i, alpha, beta_min)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]