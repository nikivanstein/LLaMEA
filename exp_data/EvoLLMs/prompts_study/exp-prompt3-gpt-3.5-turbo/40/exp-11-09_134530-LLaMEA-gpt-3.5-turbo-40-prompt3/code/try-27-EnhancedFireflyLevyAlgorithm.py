import numpy as np

class EnhancedFireflyLevyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.5

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.gamma) * np.math.sin(np.pi * self.gamma / 2) / (np.math.gamma((1 + self.gamma) / 2) * self.gamma * 2 ** ((self.gamma - 1) / 2))) ** (1 / self.gamma)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / self.gamma)
        return step

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = [func(individual) for individual in population]

        for _ in range(self.budget):
            for i in range(population_size):
                for j in range(population_size):
                    if fitness[i] > fitness[j]:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r ** 2)
                        step = self.levy_flight()
                        mutation_rate = 1 / (1 + r)
                        population[i] += beta * (population[j] - population[i]) + self.alpha * mutation_rate * step
                        population[i] = np.clip(population[i], -5.0, 5.0)
                        fitness[i] = func(population[i])

            alpha_min = 0.1
            alpha_max = 0.5
            self.alpha = alpha_min + (alpha_max - alpha_min) * (_ / self.budget)

            population_size = int(40 + 10 * np.sin(_ / self.budget * np.pi))
            if population_size > len(population):
                new_population = np.random.uniform(-5.0, 5.0, (population_size - len(population), self.dim))
                population = np.vstack([population, new_population])
                fitness.extend([func(individual) for individual in new_population])

        best_index = np.argmin(fitness)
        return population[best_index]