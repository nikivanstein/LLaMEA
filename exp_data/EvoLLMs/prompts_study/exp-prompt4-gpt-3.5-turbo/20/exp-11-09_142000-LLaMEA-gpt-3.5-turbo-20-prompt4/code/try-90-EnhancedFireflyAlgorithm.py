import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def attractiveness(self, light_intensity, distance):
        beta = 1
        return light_intensity / (1 + beta * distance)

    def move_firefly(self, idx, iter_count, max_iter):
        alpha_min = 0.2
        alpha_max = 0.9
        alpha = alpha_min + (alpha_max - alpha_min) * (1 - iter_count / max_iter)  # Dynamic alpha value
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                self.population[idx] += alpha * np.exp(-0.2 * distance) * (self.population[i] - self.population[idx])

    def __call__(self, func):
        max_iter = 100  # Define maximum iterations
        for iter_count in range(max_iter):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i, iter_count, max_iter)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]