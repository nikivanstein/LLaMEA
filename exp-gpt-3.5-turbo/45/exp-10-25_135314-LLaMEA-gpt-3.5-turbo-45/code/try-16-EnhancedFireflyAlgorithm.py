import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def attractiveness(self, light_intensity, distance):
        return self.beta0 * np.exp(-self.gamma * distance**2)

    def levy_flight(self, scale=0.1):
        sigma = (np.math.gamma(1 + scale) * np.math.sin(np.pi * scale / 2) / np.math.gamma((1 + scale) / 2) * scale ** ((scale - 1) / 2)) ** (1 / scale)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        levy = u / np.power(np.abs(v), 1 / scale)
        return levy

    def move_firefly(self, current, best, attractiveness):
        step = self.alpha * self.levy_flight()  # Incorporating Levy flights for exploration
        return current + attractiveness * (best - current) + step

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if light_intensities[j] > light_intensities[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness_ij = self.attractiveness(light_intensities[j], distance)
                        population[i] = self.move_firefly(population[i], population[j], attractiveness_ij)
                        light_intensities[i] = func(population[i])

        return population[np.argmin(light_intensities)]