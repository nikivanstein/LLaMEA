import numpy as np

class GlowwormSwarmOptimization:
    def __init__(self, budget, dim, rho=0.5, gamma=0.5, step_size=0.1):
        self.budget = budget
        self.dim = dim
        self.rho = rho
        self.gamma = gamma
        self.step_size = step_size

    def calculate_luciferin(self, light_intensity):
        return 1 / (1 + light_intensity)

    def move_glowworm(self, current, best, luciferin):
        step = self.step_size * (np.random.rand(self.dim) - 0.5)
        return current + luciferin * (best - current) + step

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])
        luciferins = np.array([self.calculate_luciferin(intensity) for intensity in light_intensities])

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if luciferins[j] > luciferins[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        if distance != 0:
                            current_luciferin = self.calculate_luciferin(light_intensities[i])
                            luciferins[i] = current_luciferin + self.rho * (current_luciferin - luciferins[i])
                            population[i] = self.move_glowworm(population[i], population[j], luciferins[i])
                            light_intensities[i] = func(population[i])

        return population[np.argmin(light_intensities)]