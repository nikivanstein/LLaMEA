import numpy as np

class RefinedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = None

    def levy_flight(self, size):
        sigma = (np.math.gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2) / (np.math.gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1 / 1.5)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.absolute(v) ** (1 / 1.5)
        return step

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        def attractiveness(i, j):
            r = np.linalg.norm(self.population[i] - self.population[j])
            return 1 / (1 + r)

        def move_fireflies(alpha=1.0, beta=1.0, gamma=0.5):
            new_population = np.copy(self.population)
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        step = self.levy_flight(self.dim)
                        new_population[i] += alpha * np.exp(-beta * np.linalg.norm(self.population[j] - self.population[i]) ** 2) * (self.population[j] - self.population[i]) + gamma * step
            self.population = new_population

        self.population = initialize_population()
        for _ in range(self.budget):
            move_fireflies()
        best_solution = min(self.population, key=lambda x: func(x))
        return best_solution