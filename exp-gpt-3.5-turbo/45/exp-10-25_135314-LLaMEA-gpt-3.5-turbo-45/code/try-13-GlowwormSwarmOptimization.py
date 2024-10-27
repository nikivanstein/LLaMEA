import numpy as np

class GlowwormSwarmOptimization:
    def __init__(self, budget, dim, step_size=0.1, neighborhood_radius=0.5):
        self.budget = budget
        self.dim = dim
        self.step_size = step_size
        self.neighborhood_radius = neighborhood_radius

    def distance(self, x, y):
        return np.linalg.norm(x - y)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if light_intensities[j] > light_intensities[i] and self.distance(population[i], population[j]) < self.neighborhood_radius:
                        step = self.step_size * (np.random.rand(self.dim) - 0.5)
                        population[i] += step
                        light_intensities[i] = func(population[i])

        return population[np.argmin(light_intensities)]