import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.2
        self.beta_min = 0.2
        self.beta_max = 2.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def attractiveness(self, r):
        return self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.alpha * r**2)

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = self.attractiveness(r)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + np.random.uniform(-1, 1, self.dim)

        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.population_size, self.dim))

        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)

        best_idx = np.argmin([func(individual) for individual in fireflies])
        return fireflies[best_idx]