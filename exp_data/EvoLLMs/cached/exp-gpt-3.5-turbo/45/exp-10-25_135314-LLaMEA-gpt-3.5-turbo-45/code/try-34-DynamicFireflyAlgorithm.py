import numpy as np

class DynamicFireflyAlgorithm(FireflyAlgorithm):
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            self.alpha = 0.2 * np.exp(-0.1 * _)  # Dynamic alpha
            self.gamma = 1.0 * np.exp(-0.1 * _)  # Dynamic gamma
            for i in range(self.budget):
                for j in range(self.budget):
                    if light_intensities[j] > light_intensities[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness_ij = self.attractiveness(light_intensities[j], distance)
                        population[i] = self.move_firefly(population[i], population[j], attractiveness_ij)
                        light_intensities[i] = func(population[i])

        return population[np.argmin(light_intensities)]