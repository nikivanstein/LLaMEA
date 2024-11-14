import numpy as np

class ImprovedFireflyLevyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.5
        self.min_population = 10
        self.max_population = 100

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.gamma) * np.math.sin(np.pi * self.gamma / 2) / (np.math.gamma((1 + self.gamma) / 2) * self.gamma * 2 ** ((self.gamma - 1) / 2))) ** (1 / self.gamma)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / self.gamma)
        return step

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = [func(individual) for individual in population]
        
        for _ in range(self.budget):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[i] > fitness[j]:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r ** 2)
                        step = self.levy_flight()
                        population[i] += beta * (population[j] - population[i]) + self.alpha * step
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])
            
            diversity = np.mean(np.std(population, axis=0))
            if diversity < 0.1 and self.population_size > self.min_population:
                self.population_size = int(0.9 * self.population_size)
            elif diversity > 0.5 and self.population_size < self.max_population:
                self.population_size = int(1.1 * self.population_size)
        
        best_index = np.argmin(fitness)
        return population[best_index]