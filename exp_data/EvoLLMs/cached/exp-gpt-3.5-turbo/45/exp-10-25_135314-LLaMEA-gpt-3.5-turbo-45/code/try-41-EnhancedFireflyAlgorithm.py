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

    def move_firefly(self, current, best, attractiveness, step_size):
        step = self.alpha * step_size * (np.random.standard_cauchy(self.dim) / np.abs(np.random.standard_cauchy(self.dim)))
        return current + attractiveness * (best - current) + step

    def levy_flight(self):
        return np.random.standard_cauchy()

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if light_intensities[j] > light_intensities[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness_ij = self.attractiveness(light_intensities[j], distance)
                        step_size = self.levy_flight()
                        population[i] = self.move_firefly(population[i], population[j], attractiveness_ij, step_size)
                        light_intensities[i] = func(population[i])

        return population[np.argmin(light_intensities)]