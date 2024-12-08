import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta_min=0.2, beta_max=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max

    def initialize_fireflies(self):
        return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

    def move_firefly(self, firefly, best_firefly):
        beta = self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.alpha * np.square(np.linalg.norm(firefly - best_firefly)))
        return firefly + beta * (best_firefly - firefly) + np.random.normal(0, 1, size=self.dim)

    def __call__(self, func):
        fireflies = self.initialize_fireflies()
        best_firefly = fireflies[np.argmin([func(individual) for individual in fireflies])]

        for _ in range(self.budget):
            for i in range(self.budget):
                new_firefly = self.move_firefly(fireflies[i], best_firefly)
                if func(new_firefly) < func(fireflies[i]):
                    fireflies[i] = new_firefly
            best_firefly = fireflies[np.argmin([func(individual) for individual in fireflies])]

        return best_firefly