import numpy as np

class DynamicFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, light_intensity, distance):
        return self.beta0 * np.exp(-self.gamma * distance**2)

    def move_firefly(self, current, best, attractiveness):
        step = self.alpha * (np.random.rand(self.dim) - 0.5)
        return current + attractiveness * (best - current) + step

    def adapt_parameters(self, iteration, max_iterations):
        self.alpha = 0.1 + 0.9 * (1 - iteration / max_iterations)
        self.beta0 = 0.9 + 0.1 * (1 - iteration / max_iterations)
        self.gamma = 1.0 - 0.5 * (1 - iteration / max_iterations)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])

        for iter in range(self.budget):
            self.adapt_parameters(iter, self.budget)
            for i in range(self.budget):
                for j in range(self.budget):
                    if light_intensities[j] > light_intensities[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness_ij = self.attractiveness(light_intensities[j], distance)
                        population[i] = self.move_firefly(population[i], population[j], attractiveness_ij)
                        light_intensities[i] = func(population[i])

        return population[np.argmin(light_intensities)]