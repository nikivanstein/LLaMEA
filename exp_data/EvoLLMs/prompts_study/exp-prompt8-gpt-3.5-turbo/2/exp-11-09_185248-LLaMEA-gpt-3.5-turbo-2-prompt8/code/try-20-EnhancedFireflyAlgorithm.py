import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.2
        self.beta0 = 1.0
        self.gamma = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def chaotic_map(self, x):
        return 3.9 * x * (1 - x)

    def levy_flight(self):
        sigma = (np.math.gamma(1 + self.beta0) * np.sin(np.pi * self.beta0 / 2) / (np.math.gamma((1 + self.beta0) / 2) * self.beta0 * 2 ** ((self.beta0 - 1) / 2))) ** (1 / self.beta0)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / self.beta0)
        return step

    def __call__(self, func):
        population = self.initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(population[j]) < func(population[i]):
                        population[i] += self.alpha * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])) * self.levy_flight() * self.chaotic_map(np.random.random())  # Introducing chaotic dynamics
                        evaluations += 1
                        if evaluations >= self.budget:
                            break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution