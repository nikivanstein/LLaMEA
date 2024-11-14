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
    
    def dynamic_alpha(self, iteration, max_iterations, initial_alpha=0.5):
        return initial_alpha * (1 - iteration / max_iterations)
    
    def __call__(self, func):
        max_iterations = self.budget
        for t in range(max_iterations):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        alpha = self.dynamic_alpha(t, max_iterations)
                        self.move_firefly(i, alpha=alpha)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]