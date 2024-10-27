import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(10, self.dim))
        intensity = np.zeros(10)
        best_solution = population[0]
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            for i in range(10):
                for j in range(10):
                    if func(population[j]) < func(population[i]):
                        intensity[i] += np.exp(-0.1 * np.linalg.norm(population[i] - population[j]))

                population[i] += intensity[i] * self.levy_flight()
                if func(population[i]) < best_fitness:
                    best_solution = population[i]
                    best_fitness = func(best_solution)

            if np.random.uniform() < 0.35:
                line_direction = np.random.uniform(-1, 1, size=(self.dim,))
                line_direction /= np.linalg.norm(line_direction)
                line_length = np.random.uniform(0.1, 1.0)
                line_point = best_solution + line_length * line_direction
                line_fitness = func(line_point)
                if line_fitness < best_fitness:
                    best_solution = line_point
                    best_fitness = line_fitness

        return best_solution